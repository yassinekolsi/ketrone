from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console

from src.config import RAW_DIR
from src.io import read_jsonl, write_jsonl
from src.vector_ops.embeddings import cosine_similarity


app = typer.Typer(help="Audit or merge near-duplicate topic labels by vector proximity.")
console = Console()


def topic_centroids(rows: list[dict]) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        name = row.get("canonical_name")
        embedding = row.get("embedding")
        if name and isinstance(embedding, list):
            grouped[name].append(embedding)
    centroids: dict[str, list[float]] = {}
    for name, vectors in grouped.items():
        dim = len(vectors[0])
        centroid = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(dim)]
        centroids[name] = centroid
    return centroids


def duplicate_pairs(rows: list[dict], threshold: float) -> list[dict]:
    centroids = topic_centroids(rows)
    names = sorted(centroids)
    pairs: list[dict] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            score = cosine_similarity(centroids[left], centroids[right])
            if score >= threshold:
                canonical = min(left, right, key=len)
                duplicate = right if canonical == left else left
                pairs.append({"canonical": canonical, "duplicate": duplicate, "score": score})
    return sorted(pairs, key=lambda row: row["score"], reverse=True)


@app.command()
def main(
    topics_path: Path = typer.Option(RAW_DIR / "topics.embedded.jsonl"),
    output_path: Path = typer.Option(RAW_DIR / "topic_merge_audit.jsonl"),
    threshold: float = 0.88,
    apply: bool = typer.Option(False, "--apply", help="Write topics.merged.jsonl with duplicate canonical names replaced."),
) -> None:
    topics = read_jsonl(topics_path)
    pairs = duplicate_pairs(topics, threshold)
    write_jsonl(output_path, pairs)
    console.print(f"[green]wrote {len(pairs)} candidate merges to {output_path}[/]")
    if apply and pairs:
        mapping = {pair["duplicate"]: pair["canonical"] for pair in pairs}
        merged = []
        for topic in topics:
            updated = dict(topic)
            if updated.get("canonical_name") in mapping:
                updated["canonical_name"] = mapping[updated["canonical_name"]]
            merged.append(updated)
        write_jsonl(topics_path.with_name("topics.merged.jsonl"), merged)
        console.print("[yellow]wrote topics.merged.jsonl; inspect before graph loading[/]")


if __name__ == "__main__":
    app()
