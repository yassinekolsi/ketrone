from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.console import Console

from src.config import RAW_DIR
from src.io import read_jsonl, write_jsonl
from src.models import Chunk


app = typer.Typer(help="Create semantically coherent chunks from document Markdown.")
console = Console()
TOKEN_RE = re.compile(r"\S+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def approx_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def split_markdown_sections(markdown: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in (markdown or "").splitlines():
        heading = HEADING_RE.match(line.strip())
        if heading:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = heading.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def window_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= max_tokens:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def chunk_document(
    document: dict,
    *,
    language: str,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    content_key = "content_ar" if language == "ar" else "content_en"
    markdown = document.get(content_key) or document.get({"ar": "contentAr", "en": "contentEn"}[language]) or ""
    sections = split_markdown_sections(markdown)
    chunks: list[Chunk] = []
    buffer = ""
    heading_path: str | None = None

    def flush() -> None:
        nonlocal buffer, heading_path
        if not buffer.strip():
            return
        for piece in window_text(buffer, max_tokens, overlap_tokens):
            chunks.append(
                Chunk(
                    id=f"{document['id']}:{language}:{len(chunks):04d}",
                    document_id=document["id"],
                    language=language,
                    text=piece,
                    order=len(chunks),
                    heading_path=heading_path,
                )
            )
        buffer = ""

    for heading, section_text in sections:
        section_tokens = approx_tokens(section_text)
        if buffer and approx_tokens(buffer) + section_tokens > max_tokens:
            flush()
        heading_path = heading or heading_path
        if section_tokens > max_tokens:
            flush()
            for piece in window_text(section_text, max_tokens, overlap_tokens):
                chunks.append(
                    Chunk(
                        id=f"{document['id']}:{language}:{len(chunks):04d}",
                        document_id=document["id"],
                        language=language,
                        text=piece,
                        order=len(chunks),
                        heading_path=heading_path,
                    )
                )
        else:
            buffer = f"{buffer}\n\n{section_text}".strip()
    flush()
    return chunks


def chunk_documents(documents: list[dict], *, max_tokens: int = 800, overlap_tokens: int = 100) -> list[dict]:
    rows: list[dict] = []
    for document in documents:
        for language in ["ar", "en"]:
            content = document.get(f"content_{language}") or document.get({"ar": "contentAr", "en": "contentEn"}[language])
            if not content:
                continue
            rows.extend(chunk.to_dict() for chunk in chunk_document(document, language=language, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return rows


@app.command()
def main(
    input_path: Path = typer.Option(RAW_DIR / "documents.jsonl"),
    output_path: Path = typer.Option(RAW_DIR / "chunks.jsonl"),
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> None:
    documents = read_jsonl(input_path)
    chunks = chunk_documents(documents, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    write_jsonl(output_path, chunks)
    console.print(f"[green]wrote {len(chunks)} chunks to {output_path}[/]")


if __name__ == "__main__":
    app()
