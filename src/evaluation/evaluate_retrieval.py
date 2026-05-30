from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from src.search_client import LocalHybridSearch


app = typer.Typer(help="Evaluate hybrid retrieval against labeled known-answer queries.")
console = Console(highlight=False)


DEFAULT_QUERIES = [
    {
        "id": "health_specialties",
        "query": "Which decree abolished the Higher Institute of Health Specialties?",
        "expected_document_ids": ["rd2026060"],
        "notes": "Should retrieve Royal Decree 60/2026.",
    },
    {
        "id": "sports_entities",
        "query": "What law regulates sports entities and sports clubs?",
        "expected_document_ids": ["rd2026059"],
        "notes": "Should retrieve Royal Decree 59/2026 issuing the Sports Entities Law.",
    },
    {
        "id": "urban_planning",
        "query": "Which decree issued the Urban Planning Law?",
        "expected_document_ids": ["rd2026058"],
        "notes": "Should retrieve Royal Decree 58/2026.",
    },
    {
        "id": "postal_sector",
        "query": "Which decree concerns the postal sector and postal services?",
        "expected_document_ids": ["rd2026057"],
        "notes": "Should retrieve Royal Decree 57/2026.",
    },
    {
        "id": "judicial_enforcement_commerce",
        "query": "Which ministerial decision gives judicial enforcement status to Ministry of Commerce officers?",
        "expected_document_ids": ["mjla20260050"],
        "notes": "Should retrieve Ministerial Decision 50/2026.",
    },
    {
        "id": "university_branch_sur",
        "query": "Which decision licensed a branch of A'Sharqiya University in Sur?",
        "expected_document_ids": ["education20260104"],
        "notes": "Should retrieve Ministerial Decision 104/2026.",
    },
    {
        "id": "adversarial_mars_colony",
        "query": "Which Omani decree regulates mining colonies on Mars?",
        "expected_document_ids": [],
        "notes": "Negative-control query. A healthy system should report no known expected document.",
    },
]


@dataclass
class EvalResult:
    query_id: str
    query: str
    expected_document_ids: list[str]
    retrieved_document_ids: list[str]
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    top_score: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "expected_document_ids": self.expected_document_ids,
            "retrieved_document_ids": self.retrieved_document_ids,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "reciprocal_rank": self.reciprocal_rank,
            "top_score": self.top_score,
            "notes": self.notes,
        }


def load_queries(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_QUERIES
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(data_dir: Path, queries: list[dict[str, Any]], *, top_k: int = 10, dense_weight: float = 0.35) -> list[EvalResult]:
    searcher = LocalHybridSearch(data_dir)
    results: list[EvalResult] = []
    available_docs = set(searcher.documents)
    for item in queries:
        expected = [doc_id for doc_id in item.get("expected_document_ids", []) if doc_id in available_docs]
        candidates = searcher.search(item["query"], k=top_k, dense_weight=dense_weight)
        retrieved = []
        for candidate in candidates:
            doc_id = candidate["chunk"]["document_id"]
            if doc_id not in retrieved:
                retrieved.append(doc_id)
        rank = 0
        for index, doc_id in enumerate(retrieved, start=1):
            if doc_id in expected:
                rank = index
                break
        results.append(
            EvalResult(
                query_id=item["id"],
                query=item["query"],
                expected_document_ids=expected,
                retrieved_document_ids=retrieved[:top_k],
                hit_at_1=rank == 1,
                hit_at_3=0 < rank <= 3,
                reciprocal_rank=(1.0 / rank) if rank else 0.0,
                top_score=float(candidates[0]["score"]) if candidates else 0.0,
                notes=item.get("notes", ""),
            )
        )
    return results


def summarize(results: list[EvalResult]) -> dict[str, float]:
    positive = [result for result in results if result.expected_document_ids]
    if not positive:
        return {"queries": float(len(results)), "hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0}
    return {
        "queries": float(len(results)),
        "positive_queries": float(len(positive)),
        "hit_at_1": sum(result.hit_at_1 for result in positive) / len(positive),
        "hit_at_3": sum(result.hit_at_3 for result in positive) / len(positive),
        "mrr": sum(result.reciprocal_rank for result in positive) / len(positive),
    }


def render(results: list[EvalResult], summary: dict[str, float]) -> None:
    table = Table(title="Retrieval Evaluation")
    table.add_column("Query")
    table.add_column("Expected")
    table.add_column("Top Docs")
    table.add_column("H@1")
    table.add_column("H@3")
    table.add_column("RR")
    for result in results:
        table.add_row(
            result.query_id,
            ", ".join(result.expected_document_ids) or "(none)",
            ", ".join(result.retrieved_document_ids[:3]) or "(none)",
            "yes" if result.hit_at_1 else "no",
            "yes" if result.hit_at_3 else "no",
            f"{result.reciprocal_rank:.2f}",
        )
    console.print(table)
    console.print({key: round(value, 3) for key, value in summary.items()})


@app.command()
def main(
    data_dir: Path = typer.Option(..., help="Directory containing documents/chunks/topics JSONL."),
    queries_path: Path | None = typer.Option(None, help="Optional JSON file of labeled queries."),
    output_path: Path | None = typer.Option(None, help="Write JSON evaluation results."),
    top_k: int = typer.Option(10, help="Candidate depth to evaluate."),
    dense_weight: float = typer.Option(0.35, help="Hybrid score weight for dense similarity; remainder is BM25."),
) -> None:
    queries = load_queries(queries_path)
    results = evaluate(data_dir, queries, top_k=top_k, dense_weight=dense_weight)
    summary = summarize(results)
    render(results, summary)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"summary": summary, "results": [result.to_dict() for result in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    app()
