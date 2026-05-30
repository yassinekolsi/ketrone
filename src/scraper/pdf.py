from __future__ import annotations

from pathlib import Path

import fitz


def pdf_bytes_to_markdown(content: bytes) -> str:
    """Extract text from a PDF and wrap pages as Markdown headings."""
    doc = fitz.open(stream=content, filetype="pdf")
    pages: list[str] = []
    for index, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"## Page {index}\n\n{text}")
    return "\n\n".join(pages).strip() + "\n"


def pdf_file_to_markdown(path: Path) -> str:
    return pdf_bytes_to_markdown(path.read_bytes())
