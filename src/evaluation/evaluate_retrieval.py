from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from src.io import read_jsonl
from src.search_client import LocalHybridSearch, dedupe_by_document


app = typer.Typer(help="Evaluate dense, hybrid+graph, and reranked retrieval stages.")
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
        "notes": "Negative control; excluded from positive-query aggregate metrics.",
    },
]

STAGE_DENSE = "dense_only"
STAGE_GRAPH = "hybrid_graph"
STAGE_RERANK = "hybrid_graph_rerank"


@dataclass
class EvalResult:
    stage: str
    query_id: str
    query: str
    expected_document_ids: list[str]
    retrieved_document_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit_at_1: bool
    hit_at_3: bool
    top_score: float
    rerank_applied: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_queries(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_QUERIES
    return json.loads(path.read_text(encoding="utf-8"))


def document_ids(candidates: list[dict[str, Any]]) -> list[str]:
    return [candidate["chunk"]["document_id"] for candidate in dedupe_by_document(candidates)]


def score_result(
    *,
    stage: str,
    item: dict[str, Any],
    expected: list[str],
    candidates: list[dict[str, Any]],
    metric_k: int,
) -> EvalResult:
    retrieved = document_ids(candidates)
    relevant_at_k = len(set(retrieved[:metric_k]) & set(expected))
    rank = next((index for index, doc_id in enumerate(retrieved, start=1) if doc_id in expected), 0)
    return EvalResult(
        stage=stage,
        query_id=item["id"],
        query=item["query"],
        expected_document_ids=expected,
        retrieved_document_ids=retrieved,
        precision_at_k=(relevant_at_k / metric_k) if expected else 0.0,
        recall_at_k=(relevant_at_k / len(expected)) if expected else 0.0,
        reciprocal_rank=(1.0 / rank) if rank else 0.0,
        hit_at_1=rank == 1,
        hit_at_3=0 < rank <= 3,
        top_score=float(candidates[0]["score"]) if candidates else 0.0,
        rerank_applied=any(candidate.get("rerank_applied") is True for candidate in candidates),
        notes=item.get("notes", ""),
    )


def evaluate(
    data_dir: Path,
    queries: list[dict[str, Any]],
    *,
    top_k: int = 20,
    metric_k: int = 3,
    dense_weight: float = 0.35,
    run_reranker: bool = True,
) -> list[EvalResult]:
    searcher = LocalHybridSearch(data_dir)
    available_docs = set(searcher.documents)
    results: list[EvalResult] = []
    for item in queries:
        expected = [doc_id for doc_id in item.get("expected_document_ids", []) if doc_id in available_docs]
        dense_candidates = searcher.search_dense(item["query"], k=top_k)
        graph_candidates = searcher.search(item["query"], k=top_k, dense_weight=dense_weight)
        reranked_candidates = (
            searcher.rerank(item["query"], [dict(candidate) for candidate in graph_candidates], top_n=top_k)
            if run_reranker
            else [dict(candidate) for candidate in graph_candidates]
        )
        for stage, candidates in [
            (STAGE_DENSE, dense_candidates),
            (STAGE_GRAPH, graph_candidates),
            (STAGE_RERANK, reranked_candidates),
        ]:
            results.append(
                score_result(
                    stage=stage,
                    item=item,
                    expected=expected,
                    candidates=candidates,
                    metric_k=metric_k,
                )
            )
    return results


def summarize(results: list[EvalResult], *, metric_k: int = 3) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    stages = [STAGE_DENSE, STAGE_GRAPH, STAGE_RERANK]
    for stage in stages:
        stage_results = [result for result in results if result.stage == stage]
        positives = [result for result in stage_results if result.expected_document_ids]
        divisor = len(positives) or 1
        summaries.append(
            {
                "stage": stage,
                "queries": len(stage_results),
                "positive_queries": len(positives),
                f"precision_at_{metric_k}": sum(result.precision_at_k for result in positives) / divisor,
                f"recall_at_{metric_k}": sum(result.recall_at_k for result in positives) / divisor,
                "hit_at_1": sum(result.hit_at_1 for result in positives) / divisor,
                "hit_at_3": sum(result.hit_at_3 for result in positives) / divisor,
                "mrr": sum(result.reciprocal_rank for result in positives) / divisor,
                "rerank_applied_queries": sum(result.rerank_applied for result in positives),
            }
        )
    return summaries


def render(summaries: list[dict[str, Any]], *, metric_k: int) -> None:
    table = Table(title="Staged Retrieval Evaluation")
    table.add_column("Stage")
    table.add_column(f"P@{metric_k}")
    table.add_column(f"R@{metric_k}")
    table.add_column("H@1")
    table.add_column("H@3")
    table.add_column("MRR")
    table.add_column("Reranked")
    for summary in summaries:
        table.add_row(
            summary["stage"],
            f"{summary[f'precision_at_{metric_k}']:.3f}",
            f"{summary[f'recall_at_{metric_k}']:.3f}",
            f"{summary['hit_at_1']:.3f}",
            f"{summary['hit_at_3']:.3f}",
            f"{summary['mrr']:.3f}",
            f"{summary['rerank_applied_queries']}/{summary['positive_queries']}",
        )
    console.print(table)


@app.command()
def main(
    data_dir: Path = typer.Option(..., help="Directory containing documents/chunks/topics JSONL."),
    queries_path: Path | None = typer.Option(None, help="Optional JSON file of labeled queries."),
    output_path: Path | None = typer.Option(None, help="Write JSON evaluation results."),
    top_k: int = typer.Option(20, help="Candidate depth to evaluate."),
    metric_k: int = typer.Option(3, help="Cutoff used for precision and recall."),
    dense_weight: float = typer.Option(0.35, help="Dense weight in the hybrid candidate stage."),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Run the cross-encoder stage."),
) -> None:
    queries = load_queries(queries_path)
    results = evaluate(
        data_dir,
        queries,
        top_k=top_k,
        metric_k=metric_k,
        dense_weight=dense_weight,
        run_reranker=rerank,
    )
    summaries = summarize(results, metric_k=metric_k)
    render(summaries, metric_k=metric_k)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "dataset": {"documents": len(read_jsonl(data_dir / "documents.jsonl")), "queries": len(queries)},
                    "metric_k": metric_k,
                    "stages": summaries,
                    "results": [result.to_dict() for result in results],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    app()
