from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx
import typer
from bs4 import BeautifulSoup
from rich.console import Console
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

from src.config import RAW_DIR, SAMPLE_DIR, STATE_DB, ensure_directories, settings
from src.models import LegalDocument
from src.scraper.markdown import html_to_markdown
from src.scraper.parsers import (
    extract_article_html,
    extract_cross_references,
    extract_english_link,
    extract_number_from_detail_page,
    extract_page_title,
    extract_pdf_urls,
    extract_slug_from_url,
    infer_document_type,
    is_gazette_slug,
    links_to_qanoon_documents,
    parse_pdf_metadata,
    strip_html_text,
)
from src.scraper.pdf import pdf_bytes_to_markdown
from src.scraper.rest_probe import probe_required_endpoints
from src.scraper.state import CrawlState


app = typer.Typer(help="REST-first qanoon.om crawler with SQLite checkpointing.")
console = Console()

USER_AGENTS = [
    settings.user_agent,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]


class QanoonCrawler:
    def __init__(self, *, state_path: Path = STATE_DB, output_dir: Path = RAW_DIR):
        ensure_directories()
        self.state = CrawlState(state_path)
        self.output_dir = output_dir
        self.markdown_dir = output_dir / "markdown"
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = output_dir / "documents.jsonl"
        self.headers = {
            "User-Agent": settings.user_agent,
            "Accept": "application/json,text/html,application/xhtml+xml",
            "Accept-Language": "ar,en;q=0.9",
        }
        self.client = httpx.Client(headers=self.headers, timeout=settings.request_timeout, follow_redirects=True)

    def close(self) -> None:
        self.client.close()
        self.state.close()

    def pace(self) -> None:
        time.sleep(random.uniform(settings.min_delay_seconds, settings.max_delay_seconds))

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.pace()
        request_headers = {**self.headers, "User-Agent": random.choice(USER_AGENTS)}
        if kwargs.get("headers"):
            request_headers.update(kwargs.pop("headers"))
        response = self.client.get(url, headers=request_headers, **kwargs)
        response.raise_for_status()
        return response

    def rest_posts(self, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
        yielded = 0
        page = int(self.state.get_meta("qanoon_rest_page", "1") or "1")
        total_pages: int | None = None
        while True:
            response = self.get(
                settings.qanoon_api_url,
                params={
                    "per_page": settings.per_page,
                    "page": page,
                    "_fields": "id,date,modified,slug,link,title,content,categories",
                },
            )
            total_pages = total_pages or int(response.headers.get("X-WP-TotalPages", "0") or "0")
            posts = response.json()
            if not posts:
                break
            for post in posts:
                yield post
                yielded += 1
                if limit and yielded >= limit:
                    self.state.set_meta("qanoon_rest_page", str(page))
                    return
            self.state.set_meta("qanoon_rest_page", str(page + 1))
            if total_pages and page >= total_pages:
                break
            page += 1

    def sitemap_urls(self, *, limit: int | None = None) -> Iterable[str]:
        yielded = 0
        sitemap_index = self.get("https://qanoon.om/wp-sitemap.xml").text
        root = ElementTree.fromstring(sitemap_index)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall(".//sm:loc", ns):
            sitemap_url = loc.text or ""
            if "wp-sitemap-posts-post" not in sitemap_url:
                continue
            page_xml = self.get(sitemap_url).text
            page_root = ElementTree.fromstring(page_xml)
            for url_loc in page_root.findall(".//sm:loc", ns):
                url = url_loc.text
                if not url:
                    continue
                yield url
                yielded += 1
                if limit and yielded >= limit:
                    return

    def post_from_url(self, url: str) -> dict[str, Any]:
        slug = extract_slug_from_url(url)
        if not slug:
            raise ValueError(f"Cannot extract post slug from {url}")
        response = self.get(
            settings.qanoon_api_url,
            params={"slug": slug, "_fields": "id,date,modified,slug,link,title,content,categories"},
        )
        posts = response.json()
        if not posts:
            raise ValueError(f"No REST post found for {url}")
        return posts[0]

    def discover(self, *, limit: int | None = None, force_sitemap: bool = False) -> list[dict[str, Any]]:
        probes = probe_required_endpoints()
        use_rest = not force_sitemap and all(ok for ok, _detail in probes.values())
        console.print({"rest_probe": {name: detail for name, (_ok, detail) in probes.items()}, "use_rest": use_rest})

        posts: list[dict[str, Any]] = []
        if use_rest:
            for post in self.rest_posts(limit=limit):
                posts.append(post)
                self._checkpoint_post(post)
        else:
            for url in self.sitemap_urls(limit=limit):
                try:
                    post = self.post_from_url(url)
                except Exception as exc:
                    console.print(f"[yellow]fallback URL failed:[/] {url} {exc}")
                    continue
                posts.append(post)
                self._checkpoint_post(post)
        return posts

    def _checkpoint_post(self, post: dict[str, Any]) -> None:
        slug = post.get("slug")
        if not slug:
            return
        content_html = post.get("content", {}).get("rendered", "")
        english_url = extract_english_link(content_html)
        self.state.upsert_discovered(
            slug,
            post.get("id"),
            post.get("link") or f"https://qanoon.om/p/{slug}/",
            is_gazette=is_gazette_slug(slug),
            english_url=english_url,
        )

    def _fetch_english(self, english_url: str | None) -> tuple[str | None, str | None, str | None]:
        if not english_url:
            return None, None, None
        response = self.get(english_url, headers={**self.headers, "Accept": "text/html"})
        detail_html = response.text
        title = extract_page_title(detail_html)
        article_html = extract_article_html(detail_html)
        pdf_urls = extract_pdf_urls(article_html)
        return title, html_to_markdown(article_html), pdf_urls[0] if pdf_urls else None

    def _fallback_pdf_markdown(self, pdf_url: str | None) -> str | None:
        if not pdf_url:
            return None
        try:
            response = self.get(pdf_url, headers={**self.headers, "Accept": "application/pdf"})
        except Exception as exc:
            console.print(f"[yellow]PDF fallback failed:[/] {pdf_url} {exc}")
            return None
        return pdf_bytes_to_markdown(response.content)

    def build_document(self, post: dict[str, Any], *, pdf_fallback: bool = False) -> LegalDocument | None:
        slug = post["slug"]
        content_html = post.get("content", {}).get("rendered", "")
        if is_gazette_slug(slug):
            for linked_url in links_to_qanoon_documents(content_html):
                linked_slug = extract_slug_from_url(linked_url)
                if linked_slug:
                    self.state.upsert_discovered(linked_slug, None, linked_url)
            self.state.mark_status(slug, "skipped_gazette")
            return None

        detail_html = self.get(post["link"], headers={**self.headers, "Accept": "text/html"}).text
        number = extract_number_from_detail_page(detail_html)
        pdf_urls = extract_pdf_urls(content_html)
        pdf_url_ar = pdf_urls[0] if pdf_urls else None
        pdf_meta = parse_pdf_metadata(pdf_url_ar)
        content_ar = html_to_markdown(content_html)
        if pdf_fallback and len(content_ar.strip()) < 80:
            content_ar = self._fallback_pdf_markdown(pdf_url_ar) or content_ar

        english_url = extract_english_link(content_html)
        title_en, content_en, pdf_url_en = self._fetch_english(english_url)

        document = LegalDocument(
            id=slug,
            slug=slug,
            source_url=post.get("link") or "",
            title_ar=strip_html_text(post.get("title", {}).get("rendered")),
            title_en=title_en,
            date=post.get("date"),
            document_type=infer_document_type(slug, pdf_url_ar),
            number=number,
            issuer_code=pdf_meta.get("issuer_code"),
            pdf_url_ar=pdf_url_ar,
            pdf_url_en=pdf_url_en,
            english_url=english_url,
            content_ar=content_ar,
            content_en=content_en,
            categories=post.get("categories", []),
            cross_references=extract_cross_references(content_html, slug),
            raw={
                "wordpress_id": post.get("id"),
                "pdf_metadata": pdf_meta,
            },
        )
        self.state.mark_status(slug, "done")
        return document

    def write_document(self, document: LegalDocument, *, sample: bool = False) -> None:
        target_dir = SAMPLE_DIR if sample else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir = target_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = target_dir / "documents.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
        if document.content_ar:
            (markdown_dir / f"{document.slug}.ar.md").write_text(document.content_ar, encoding="utf-8")
        if document.content_en:
            (markdown_dir / f"{document.slug}.en.md").write_text(document.content_en, encoding="utf-8")

    def crawl(self, *, limit: int | None = None, sample: bool = False, force_sitemap: bool = False, pdf_fallback: bool = False) -> int:
        count = 0
        posts = self.discover(limit=limit, force_sitemap=force_sitemap)
        for post in tqdm(posts, desc="documents"):
            try:
                document = self.build_document(post, pdf_fallback=pdf_fallback)
            except Exception as exc:
                slug = post.get("slug", "unknown")
                self.state.mark_status(slug, "failed", str(exc))
                console.print(f"[red]failed {slug}:[/] {exc}")
                continue
            if document is None:
                continue
            self.write_document(document, sample=sample)
            count += 1
        return count


@app.command()
def probe() -> None:
    """Run the day-1 REST smoke test for qanoon.om and decree.om."""
    for name, (ok, detail) in probe_required_endpoints().items():
        console.print(f"{name}: {'REST OK' if ok else 'FALLBACK REQUIRED'} - {detail}")


@app.command()
def crawl(
    limit: int | None = typer.Option(None, help="Maximum qanoon posts to inspect."),
    sample: bool = typer.Option(False, help="Write to data/sample_output instead of data/raw."),
    output_dir: Path = typer.Option(RAW_DIR, help="Output directory when --sample is not used."),
    state_path: Path = typer.Option(STATE_DB, help="SQLite checkpoint path."),
    force_sitemap: bool = typer.Option(False, help="Force sitemap/HTML fallback discovery."),
    pdf_fallback: bool = typer.Option(False, help="Download PDFs when HTML content is too short."),
) -> None:
    crawler = QanoonCrawler(state_path=state_path, output_dir=output_dir)
    try:
        count = crawler.crawl(limit=limit, sample=sample, force_sitemap=force_sitemap, pdf_fallback=pdf_fallback)
    finally:
        crawler.close()
    console.print(f"[green]wrote {count} documents[/]")


if __name__ == "__main__":
    app()
