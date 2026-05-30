from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import typer
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from src.config import RAW_DIR, SAMPLE_DIR
from src.io import read_jsonl
from src.llm_agents.llm_client import complete_text
from src.vector_ops.embeddings import Embedder, cosine_similarity


app = typer.Typer(help="Hybrid GraphRAG search client.")
console = Console(highlight=False)
TOKEN_RE = re.compile(r"\w+|[\u0600-\u06FF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


class LocalHybridSearch:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents = {row["id"]: row for row in read_jsonl(data_dir / "documents.jsonl")}
        self.chunks = read_jsonl(data_dir / "chunks.embedded.jsonl") or read_jsonl(data_dir / "chunks.jsonl")
        self.topics = read_jsonl(data_dir / "topics.embedded.jsonl") or read_jsonl(data_dir / "topics.jsonl")
        self.topics_by_doc: dict[str, list[dict]] = defaultdict(list)
        for topic in self.topics:
            self.topics_by_doc[topic["document_id"]].append(topic)
        self.embedder = Embedder()
        self.bm25 = BM25Okapi([tokenize(self.index_text(chunk)) for chunk in self.chunks]) if self.chunks else None

    def index_text(self, chunk: dict[str, Any]) -> str:
        document = self.documents.get(chunk["document_id"], {})
        topics = self.topics_by_doc.get(chunk["document_id"], [])
        topic_text = " ".join(
            str(topic.get("canonical_name") or topic.get("name") or "")
            for topic in topics
        )
        metadata = " ".join(
            str(document.get(key) or "")
            for key in [
                "title_en",
                "title_ar",
                "titleEn",
                "titleAr",
                "document_type",
                "type",
                "number",
                "issuer_code",
                "issuerCode",
            ]
        )
        return f"{chunk.get('text', '')}\n{metadata}\n{topic_text}"

    def dense_scores(self, query: str) -> list[float]:
        if not self.chunks:
            return []
        query_vector = self.embedder.encode([query], kind="query")[0]
        scores = []
        for chunk in self.chunks:
            embedding = chunk.get("embedding")
            scores.append(cosine_similarity(query_vector, embedding) if isinstance(embedding, list) else 0.0)
        return scores

    def bm25_scores(self, query: str) -> list[float]:
        if not self.bm25:
            return []
        return [float(score) for score in self.bm25.get_scores(tokenize(query))]

    @staticmethod
    def normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        arr = np.asarray(scores, dtype=float)
        min_score = float(arr.min())
        max_score = float(arr.max())
        if max_score == min_score:
            return [0.0 for _ in scores]
        return [float((score - min_score) / (max_score - min_score)) for score in scores]

    def search(self, query: str, *, k: int = 50, dense_weight: float = 0.35) -> list[dict[str, Any]]:
        dense = self.normalize_scores(self.dense_scores(query))
        sparse = self.normalize_scores(self.bm25_scores(query))
        candidates: list[dict[str, Any]] = []
        for index, chunk in enumerate(self.chunks):
            dense_score = dense[index] if index < len(dense) else 0.0
            sparse_score = sparse[index] if index < len(sparse) else 0.0
            score = dense_weight * dense_score + (1.0 - dense_weight) * sparse_score
            if score <= 0:
                continue
            candidates.append(
                {
                    "chunk": chunk,
                    "score": score,
                    "dense_score": dense_score,
                    "bm25_score": sparse_score,
                    "expanded_text": self.expand_candidate(chunk),
                }
            )
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:k]

    def expand_candidate(self, chunk: dict[str, Any]) -> str:
        document = self.documents.get(chunk["document_id"], {})
        topics = self.topics_by_doc.get(chunk["document_id"], [])
        topic_text = ", ".join(sorted({topic.get("canonical_name") or topic.get("name") for topic in topics if topic.get("name")}))
        refs = document.get("cross_references", [])
        ref_text = "; ".join(
            f"{ref.get('rel_type')} {ref.get('anchor_text')}"
            for ref in refs[:5]
            if ref.get("rel_type") and ref.get("anchor_text")
        )
        title = document.get("title_en") or document.get("title_ar") or document.get("titleEn") or document.get("titleAr")
        metadata = (
            f"Document: {title}\n"
            f"Date: {document.get('date')}\n"
            f"Type: {document.get('document_type') or document.get('type')}\n"
            f"Number: {document.get('number')}\n"
            f"Topics: {topic_text}\n"
            f"References: {ref_text}"
        )
        return f"{chunk.get('text', '')}\n\n--- Graph Context ---\n{metadata}"

    def rerank(self, query: str, candidates: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("BAAI/bge-reranker-base")
            pairs = [(query, candidate["expanded_text"]) for candidate in candidates]
            scores = model.predict(pairs)
            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)
            return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:top_n]
        except Exception as exc:
            console.print(f"[yellow]reranker unavailable, using hybrid scores:[/] {exc}")
            return candidates[:top_n]

    def synthesize(self, query: str, candidates: list[dict[str, Any]]) -> str:
        context_blocks = []
        for index, candidate in enumerate(candidates, start=1):
            chunk = candidate["chunk"]
            document = self.documents.get(chunk["document_id"], {})
            title = document.get("title_en") or document.get("title_ar") or document.get("titleEn") or document.get("titleAr")
            source = document.get("source_url") or document.get("sourceUrl")
            context_blocks.append(
                f"[{index}] {title}\nURL: {source}\n{candidate['expanded_text'][:1800]}"
            )
        context = "\n\n".join(context_blocks)
        answer = complete_text(
            "You answer legal search questions using only the supplied context. Cite bracket numbers.",
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer with a concise summary and citations.",
        )
        if answer:
            return answer
        if not candidates:
            return "No matching context was found."
        lines = ["LLM synthesis is disabled, so here is an extractive answer from the top retrieved contexts:"]
        for index, candidate in enumerate(candidates, start=1):
            chunk = candidate["chunk"]
            document = self.documents.get(chunk["document_id"], {})
            title = document.get("title_en") or document.get("title_ar") or chunk["document_id"]
            snippet = re.sub(r"\s+", " ", chunk.get("text", "")).strip()[:350]
            lines.append(f"[{index}] {title}: {snippet}")
        return "\n".join(lines)


def render_matches(candidates: list[dict[str, Any]]) -> None:
    table = Table(title="Hybrid Matches")
    table.add_column("#", justify="right")
    table.add_column("Score")
    table.add_column("Dense")
    table.add_column("BM25")
    table.add_column("Document")
    table.add_column("Topics")
    for index, candidate in enumerate(candidates, start=1):
        chunk = candidate["chunk"]
        doc_id = chunk["document_id"]
        text = candidate["expanded_text"]
        topic_line = ""
        for line in text.splitlines():
            if line.startswith("Topics:"):
                topic_line = line.removeprefix("Topics:").strip()
                break
        table.add_row(
            str(index),
            f"{candidate['score']:.3f}",
            f"{candidate['dense_score']:.3f}",
            f"{candidate['bm25_score']:.3f}",
            doc_id,
            topic_line[:60],
        )
    console.print(table)


@app.command()
def query(
    query_text: str = typer.Option(..., "--query", "-q", help="User legal search query."),
    data_dir: Path = typer.Option(RAW_DIR, help="Directory containing documents/chunks/topics JSONL."),
    top_k: int = typer.Option(50, help="Candidate generation size."),
    top_n: int = typer.Option(5, help="Final contexts to synthesize."),
    rerank: bool = typer.Option(False, help="Enable local cross-encoder reranking."),
    dense_weight: float = typer.Option(0.35, help="Hybrid score weight for dense similarity; remainder is BM25."),
) -> None:
    if not (data_dir / "documents.jsonl").exists() and (SAMPLE_DIR / "documents.jsonl").exists():
        data_dir = SAMPLE_DIR
    searcher = LocalHybridSearch(data_dir)
    candidates = searcher.search(query_text, k=top_k, dense_weight=dense_weight)
    if rerank:
        candidates = searcher.rerank(query_text, candidates, top_n=top_n)
    else:
        candidates = candidates[:top_n]
    render_matches(candidates)
    console.rule("Graph Context + Final Answer")
    console.print(searcher.synthesize(query_text, candidates), markup=False, highlight=False)


if __name__ == "__main__":
    app()
