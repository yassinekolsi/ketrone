from __future__ import annotations

from pathlib import Path

import networkx as nx
import typer
from rich.console import Console

from src.config import RAW_DIR
from src.io import read_jsonl, write_jsonl
from src.llm_agents.llm_client import complete_text


app = typer.Typer(help="Exploratory community detection over document-topic-reference graph.")
console = Console()


def summarize_community(community_id: int, topic_labels: list[str], document_titles: list[str]) -> str:
    if not topic_labels and not document_titles:
        return "Sparse community with insufficient topic or document labels."
    prompt = (
        f"Community {community_id}\n"
        f"Topic labels: {', '.join(topic_labels[:15]) or 'None'}\n"
        f"Representative documents: {'; '.join(document_titles[:8]) or 'None'}\n\n"
        "Write one concise legal-domain description for this graph community. "
        "Do not invent facts beyond the labels and titles."
    )
    summary = complete_text(
        "You label legal graph communities using only provided topic labels and document titles.",
        prompt,
    )
    if summary:
        return summary.strip()
    if topic_labels:
        return f"Community around {', '.join(topic_labels[:5])}."
    return f"Community around documents: {'; '.join(document_titles[:3])}."


@app.command()
def main(
    documents_path: Path = typer.Option(RAW_DIR / "documents.jsonl"),
    topics_path: Path = typer.Option(RAW_DIR / "topics.jsonl"),
    output_path: Path = typer.Option(RAW_DIR / "communities.jsonl"),
    summaries_path: Path | None = typer.Option(None, help="Optional JSONL output for one summary row per community."),
) -> None:
    documents = read_jsonl(documents_path)
    topics = read_jsonl(topics_path)
    graph = nx.Graph()
    document_titles = {
        f"doc:{document['id']}": document.get("title_en") or document.get("title_ar") or document["id"]
        for document in documents
    }
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
    summary_rows = []
    for community_id, nodes in enumerate(communities):
        labels = sorted(node.removeprefix("topic:") for node in nodes if node.startswith("topic:"))
        titles = sorted(document_titles[node] for node in nodes if node in document_titles)
        summary = summarize_community(community_id, labels, titles)
        summary_rows.append(
            {
                "community_id": community_id,
                "topic_labels": labels[:15],
                "representative_documents": titles[:8],
                "summary": summary,
            }
        )
        for node in nodes:
            rows.append({"node": node, "community_id": community_id, "topic_labels": labels[:10], "community_summary": summary})
    write_jsonl(output_path, rows)
    if summaries_path:
        write_jsonl(summaries_path, summary_rows)
    console.print(f"[green]wrote {len(rows)} node community assignments[/]")


if __name__ == "__main__":
    app()
