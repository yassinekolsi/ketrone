from __future__ import annotations

from bs4 import BeautifulSoup

try:
    from markdownify import markdownify as md
except ImportError:  # pragma: no cover - exercised only on minimal local Python installs.
    md = None


NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "form",
    "header",
    "footer",
    "nav",
    "[role='banner']",
    "[role='contentinfo']",
    ".sharedaddy",
    ".post-meta",
    ".nav-links",
    ".comments-wrapper",
    ".footer-nav-widgets-wrapper",
]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    for tag in soup.find_all(True):
        attrs = {}
        if tag.name == "a" and tag.get("href"):
            attrs["href"] = tag["href"]
        if tag.name == "img" and tag.get("src"):
            attrs["src"] = tag["src"]
            if tag.get("alt"):
                attrs["alt"] = tag["alt"]
        tag.attrs = attrs
    return str(soup)


def html_to_markdown(html: str) -> str:
    cleaned = clean_html(html)
    if md is None:
        markdown = fallback_markdown(cleaned)
    else:
        markdown = md(
            cleaned,
            heading_style="ATX",
            bullets="-",
            convert=["a", "p", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "table", "tr", "td", "th", "ul", "ol", "li", "blockquote"],
        )
    lines = [line.rstrip() for line in markdown.splitlines()]
    compact: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                compact.append("")
            blank = True
            continue
        compact.append(line)
        blank = False
    return "\n".join(compact).strip() + "\n"


def fallback_markdown(cleaned_html: str) -> str:
    soup = BeautifulSoup(cleaned_html or "", "html.parser")
    lines: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "tr"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            level = int(element.name[1])
            lines.append(f"{'#' * level} {text}")
        elif element.name == "li":
            lines.append(f"- {text}")
        elif element.name == "blockquote":
            lines.append(f"> {text}")
        elif element.name == "tr":
            cells = [cell.get_text(" ", strip=True) for cell in element.find_all(["th", "td"])]
            if cells:
                lines.append("| " + " | ".join(cells) + " |")
                if element.find("th"):
                    lines.append("| " + " | ".join("---" for _ in cells) + " |")
        else:
            for strong in element.find_all("strong"):
                strong.string = f"**{strong.get_text(' ', strip=True)}**"
            lines.append(element.get_text(" ", strip=True))
        lines.append("")
    return "\n".join(lines)
