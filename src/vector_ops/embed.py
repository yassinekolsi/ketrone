from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from src.config import RAW_DIR, settings
from src.io import read_jsonl, write_jsonl
from src.vector_ops.embeddings import Embedder


app = typer.Typer(help="Generate multilingual embeddings for chunks and topics.")
console = Console()


def embed_rows(rows: list[dict], *, text_key: str, kind: str, batch_size: int = 32) -> list[dict]:
    embedder = Embedder(settings.embedding_model)
    output: list[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [str(row.get(text_key) or "") for row in batch]
        vectors = embedder.encode(texts, kind=kind)
        for row, vector in zip(batch, vectors):
            updated = dict(row)
            updated["embedding"] = vector
            output.append(updated)
    return output


@app.command()
def main(
    chunks_path: Path = typer.Option(RAW_DIR / "chunks.jsonl"),
    topics_path: Path = typer.Option(RAW_DIR / "topics.jsonl"),
    output_dir: Path = typer.Option(RAW_DIR),
    batch_size: int = 32,
) -> None:
    chunks = read_jsonl(chunks_path)
    topics = read_jsonl(topics_path)
    if chunks:
        embedded_chunks = embed_rows(chunks, text_key="text", kind="passage", batch_size=batch_size)
        write_jsonl(output_dir / "chunks.embedded.jsonl", embedded_chunks)
        console.print(f"[green]embedded {len(embedded_chunks)} chunks[/]")
    if topics:
        topic_rows = [dict(row, topic_text=row.get("canonical_name") or row.get("name") or "") for row in topics]
        embedded_topics = embed_rows(topic_rows, text_key="topic_text", kind="passage", batch_size=batch_size)
        for row in embedded_topics:
            row.pop("topic_text", None)
        write_jsonl(output_dir / "topics.embedded.jsonl", embedded_topics)
        console.print(f"[green]embedded {len(embedded_topics)} topic links[/]")


if __name__ == "__main__":
    app()
