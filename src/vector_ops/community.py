from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import typer
from rich.console import Console

from src.config import RAW_DIR
from src.io import read_jsonl, write_jsonl


app = typer.Typer(help="Exploratory community detection over document-topic-reference graph.")
console = Console()


@app.command()
def main(
    documents_path: Path = typer.Option(RAW_DIR / "documents.jsonl"),
    topics_path: Path = typer.Option(RAW_DIR / "topics.jsonl"),
    output_path: Path = typer.Option(RAW_DIR / "communities.jsonl"),
) -> None:
    documents = read_jsonl(documents_path)
    topics = read_jsonl(topics_path)
    graph = nx.Graph()
    topic_names: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        doc_id = f"doc:{document['id']}"
        graph.add_node(doc_id, kind="Document")
        for ref in document.get("cross_references", []):
            if ref.get("target_slug"):
                graph.add_edge(doc_id, f"doc:{ref['target_slug']}", rel=ref.get("rel_type"))
    for topic in topics:
        doc_id = f"doc:{topic['document_id']}"
        topic_id = f"topic:{topic['canonical_name']}"
        graph.add_node(topic_id, kind="Topic")
        graph.add_edge(doc_id, topic_id, rel="HAS_TOPIC")
    communities = nx.community.louvain_communities(graph, seed=42) if graph.number_of_edges() else []
    rows = []
    for community_id, nodes in enumerate(communities):
        labels = sorted(node.removeprefix("topic:") for node in nodes if node.startswith("topic:"))
        for node in nodes:
            rows.append({"node": node, "community_id": community_id, "topic_labels": labels[:10]})
    write_jsonl(output_path, rows)
    console.print(f"[green]wrote {len(rows)} node community assignments[/]")


if __name__ == "__main__":
    app()
