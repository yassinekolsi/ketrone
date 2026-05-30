from __future__ import annotations

import html
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from src.models import CrossReference


QANOON_HOSTS = {"qanoon.om", "www.qanoon.om"}
DEGREE_HOSTS = {"decree.om", "www.decree.om"}
PDF_RE = re.compile(r"https?://data\.qanoon\.om/[^\s\"']+?\.pdf", re.IGNORECASE)
POST_SLUG_RE = re.compile(r"/(?:p/)?\d{4}/([^/?#]+)/?")
ASCII_NUMBER_BLOCK_RE = re.compile(r"\b(20\d{2})/(\d{1,4})\b\s+\b(\d{1,4})/(20\d{2})\b")

REFERENCE_KEYWORDS = ("بعد الاطلاع على", "وعلى")
REPEAL_KEYWORDS = ("يلغى", "تلغى", "إلغاء", "بالإلغاء")
AMEND_KEYWORDS = ("يعدل", "تعدل", "يستبدل", "تستبدل", "يضاف", "تضاف")


def strip_html_text(value: str | None) -> str | None:
    if value is None:
        return None
    return BeautifulSoup(html.unescape(value), "html.parser").get_text(" ", strip=True)


def extract_slug_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    match = POST_SLUG_RE.search(parsed.path)
    return match.group(1) if match else None


def is_qanoon_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.netloc in QANOON_HOSTS and "/p/" in parsed.path


def is_gazette_slug(slug: str | None) -> bool:
    return bool(slug and re.fullmatch(r"og\d+", slug))


def absolute_url(base: str, href: str | None) -> str | None:
    if not href:
        return None
    return urljoin(base, href)


def extract_english_link(content_html: str, base_url: str = "https://qanoon.om") -> str | None:
    soup = BeautifulSoup(content_html or "", "html.parser")
    decree_links: list[str] = []
    for link in soup.find_all("a", href=True):
        href = absolute_url(base_url, link["href"])
        parsed = urlparse(href or "")
        if parsed.netloc not in DEGREE_HOSTS:
            continue
        decree_links.append(href or "")
        text = link.get_text(" ", strip=True).lower()
        classes = set(link.get("class", []))
        if "decree-link" in classes or "english" in text:
            return href
    return decree_links[0] if decree_links else None


def extract_pdf_urls(content_html: str) -> list[str]:
    soup = BeautifulSoup(content_html or "", "html.parser")
    urls: list[str] = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf") and "data.qanoon.om" in href:
            urls.append(href)
    for match in PDF_RE.findall(content_html or ""):
        if match not in urls:
            urls.append(match)
    return urls


def parse_pdf_metadata(pdf_url: str | None) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {
        "family": None,
        "year": None,
        "issuer_code": None,
        "serial": None,
    }
    if not pdf_url:
        return metadata
    path = urlparse(pdf_url).path.strip("/")
    parts = path.split("/")
    if len(parts) < 3:
        return metadata
    try:
        data_index = parts.index("ar")
    except ValueError:
        data_index = -1
    if data_index >= 0 and len(parts) > data_index + 1:
        family = parts[data_index + 1]
        metadata["family"] = family
        if family == "md" and len(parts) > data_index + 2:
            metadata["issuer_code"] = parts[data_index + 2]
    elif parts[0] == "og":
        metadata["family"] = "og"

    filename = parts[-1]
    match = re.search(r"(20\d{2})-(\d+)\.pdf$", filename)
    if match:
        metadata["year"] = match.group(1)
        metadata["serial"] = str(int(match.group(2)))
    elif len(parts) >= 2 and parts[0] == "og":
        metadata["serial"] = filename.removesuffix(".pdf")
    return metadata


def infer_document_type(slug: str, pdf_url: str | None, category_names: list[str] | None = None) -> str | None:
    pdf_meta = parse_pdf_metadata(pdf_url)
    family = pdf_meta.get("family")
    if family == "rd" or slug.startswith("rd"):
        return "royal_decree"
    if family == "md":
        return "ministerial_decision"
    if family == "og" or is_gazette_slug(slug):
        return "official_gazette"
    if slug.startswith("laws") or (category_names and any("قانون" in name for name in category_names)):
        return "law"
    return family


def extract_number_from_detail_page(detail_html: str) -> str | None:
    """Parse the exact paragraph below h1.entry-title containing four number formats."""
    soup = BeautifulSoup(detail_html or "", "html.parser")
    h1 = soup.select_one("h1.entry-title")
    if not h1:
        return None
    block = h1.find_next_sibling()
    while block and isinstance(block, Tag):
        classes = set(block.get("class", []))
        if block.name == "div" and {"intro-text", "section-inner"}.issubset(classes):
            paragraph = block.find("p")
            text = paragraph.get_text(" ", strip=True) if paragraph else block.get_text(" ", strip=True)
            match = ASCII_NUMBER_BLOCK_RE.search(text)
            if match and match.group(1) == match.group(4) and match.group(2) == match.group(3):
                return f"{int(match.group(3))}/{match.group(4)}"
            return None
        if block.name == "div" and "post-meta-wrapper" in classes:
            return None
        block = block.find_next_sibling()
    return None


def classify_reference_context(context: str) -> str:
    if any(keyword in context for keyword in REPEAL_KEYWORDS):
        return "REPEALS"
    if any(keyword in context for keyword in AMEND_KEYWORDS):
        return "AMENDS"
    if any(keyword in context for keyword in REFERENCE_KEYWORDS):
        return "REFERENCES"
    return "REFERENCES"


def extract_cross_references(content_html: str, source_slug: str, base_url: str = "https://qanoon.om") -> list[CrossReference]:
    soup = BeautifulSoup(content_html or "", "html.parser")
    references: list[CrossReference] = []
    seen: set[tuple[str, str]] = set()
    for link in soup.find_all("a", href=True):
        url = absolute_url(base_url, link["href"])
        if not is_qanoon_url(url):
            continue
        target_slug = extract_slug_from_url(url)
        if not target_slug or target_slug == source_slug:
            continue
        parent = link.find_parent(["p", "li", "td", "th"]) or link
        context = parent.get_text(" ", strip=True)
        rel_type = classify_reference_context(context)
        key = (target_slug, rel_type)
        if key in seen:
            continue
        seen.add(key)
        references.append(
            CrossReference(
                source_slug=source_slug,
                target_url=url or "",
                target_slug=target_slug,
                rel_type=rel_type,
                anchor_text=link.get_text(" ", strip=True),
                context=context,
            )
        )
    return references


def links_to_qanoon_documents(content_html: str, base_url: str = "https://qanoon.om") -> list[str]:
    soup = BeautifulSoup(content_html or "", "html.parser")
    urls: list[str] = []
    for link in soup.find_all("a", href=True):
        url = absolute_url(base_url, link["href"])
        if is_qanoon_url(url) and url not in urls:
            urls.append(url)
    return urls


def extract_article_html(detail_html: str) -> str:
    soup = BeautifulSoup(detail_html or "", "html.parser")
    article = soup.select_one(".entry-content") or soup.select_one("article") or soup.body
    return str(article) if article else detail_html


def extract_page_title(detail_html: str) -> str | None:
    soup = BeautifulSoup(detail_html or "", "html.parser")
    h1 = soup.select_one("h1.entry-title") or soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else None
