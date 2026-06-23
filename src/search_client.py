from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
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

from src.config import RAW_DIR, SAMPLE_DIR, settings
from src.io import read_jsonl
from src.llm_agents.llm_client import complete_text
from src.vector_ops.embeddings import Embedder, cosine_similarity


app = typer.Typer(help="Hybrid GraphRAG search client.")
console = Console(highlight=False)
_RERANKER_LOCK = Lock()
_EMBEDDER_LOCK = Lock()
TOKEN_RE = re.compile(r"\w+|[\u0600-\u06FF]+", re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "give",
    "gives",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "who",
    "whose",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def lucene_or_query(text: str) -> str:
    tokens = [token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS]
    if not tokens:
        tokens = tokenize(text)
    # The tokens come from TOKEN_RE, so they avoid Lucene punctuation.
    return " OR ".join(tokens) if tokens else text


def normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=float)
    min_score = float(values.min())
    max_score = float(values.max())
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {key: float((value - min_score) / (max_score - min_score)) for key, value in scores.items()}


@lru_cache(maxsize=2)
def load_reranker(model_name: str):
    """Load one reusable cross-encoder per model instead of once per query."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


@dataclass
class SearchPipelineResult:
    candidates: list[dict[str, Any]]
    answer: str
    rerank_requested: bool
    rerank_applied: bool


class LocalHybridSearch:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents = {row["id"]: row for row in read_jsonl(data_dir / "documents.jsonl")}
        self.chunks = read_jsonl(data_dir / "chunks.embedded.jsonl") or read_jsonl(data_dir / "chunks.jsonl")
        self.topics = read_jsonl(data_dir / "topics.embedded.jsonl") or read_jsonl(data_dir / "topics.jsonl")
        self.topics_by_doc: dict[str, list[dict]] = defaultdict(list)
        for topic in self.topics:
            self.topics_by_doc[topic["document_id"]].append(topic)
        metadata_path = data_dir / "embedding_metadata.json"
        embedding_metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                embedding_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                embedding_metadata = {}
        self.embedder = Embedder(
            model_name=str(embedding_metadata.get("model") or settings.embedding_model),
            fallback_dim=int(embedding_metadata.get("dimension") or 384),
            force_fallback=embedding_metadata.get("backend") == "deterministic_hash",
        )
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
            weighted_score = dense_weight * dense_score + (1.0 - dense_weight) * sparse_score
            score = max(sparse_score, weighted_score)
            if score <= 0:
                continue
            candidates.append(
                {
                    "chunk": chunk,
                    "document": self.documents.get(chunk["document_id"], {}),
                    "score": score,
                    "dense_score": dense_score,
                    "bm25_score": sparse_score,
                    "expanded_text": self.expand_candidate(chunk),
                }
            )
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:k]

    def search_dense(self, query: str, *, k: int = 50) -> list[dict[str, Any]]:
        """Dense-only baseline used by the staged retrieval evaluation."""
        dense = self.normalize_scores(self.dense_scores(query))
        candidates: list[dict[str, Any]] = []
        for index, chunk in enumerate(self.chunks):
            score = dense[index] if index < len(dense) else 0.0
            if score <= 0:
                continue
            candidates.append(
                {
                    "chunk": chunk,
                    "document": self.documents.get(chunk["document_id"], {}),
                    "score": score,
                    "dense_score": score,
                    "bm25_score": 0.0,
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
            pairs = [(query, candidate["expanded_text"]) for candidate in candidates]
            if not pairs:
                return []
            # Torch model inference is serialized so concurrent API requests do not
            # race model loading or oversubscribe a small deployment instance.
            with _RERANKER_LOCK:
                model = load_reranker(settings.reranker_model)
                scores = model.predict(pairs, show_progress_bar=False)
            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)
                candidate["rerank_applied"] = True
            return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:top_n]
        except Exception as exc:
            console.print(f"[yellow]reranker unavailable, using hybrid scores:[/] {exc}")
            for candidate in candidates:
                candidate["rerank_score"] = None
                candidate["rerank_applied"] = False
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


class Neo4jHybridSearch:
    """Hybrid search backend that queries Neo4j vector/full-text indexes live."""

    def __init__(self, *, allow_self_signed: bool = False):
        from neo4j import GraphDatabase

        uri = settings.neo4j_uri
        if allow_self_signed and uri.startswith("neo4j+s://"):
            uri = uri.replace("neo4j+s://", "neo4j+ssc://", 1)
        self.driver = GraphDatabase.driver(
            uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            notifications_min_severity="OFF",
        )
        self.database = settings.neo4j_database
        self._embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            with _EMBEDDER_LOCK:
                if self._embedder is None:
                    self._embedder = Embedder()
        return self._embedder

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def vector_candidates(self, query: str, *, k: int) -> dict[str, float]:
        vector = self.embedder.encode([query], kind="query")[0]
        with self.session() as session:
            rows = session.run(
                """
                CALL db.index.vector.queryNodes('chunk_embedding', $k, $vector)
                YIELD node, score
                RETURN node.id AS chunkId, score
                """,
                k=k,
                vector=vector,
            ).data()
        return {row["chunkId"]: float(row["score"]) for row in rows if row.get("chunkId")}

    def fulltext_candidates(self, query: str, *, k: int) -> dict[str, float]:
        search_text = lucene_or_query(query)
        with self.session() as session:
            chunk_rows = session.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_text_ft', $searchText)
                YIELD node, score
                RETURN node.id AS chunkId, score
                ORDER BY score DESC
                LIMIT $k
                """,
                searchText=search_text,
                k=k,
            ).data()
            document_rows = session.run(
                """
                CALL db.index.fulltext.queryNodes('document_content_ft', $searchText)
                YIELD node, score
                MATCH (node)-[:HAS_CHUNK]->(chunk:Chunk)
                RETURN chunk.id AS chunkId, max(score) AS score
                ORDER BY score DESC
                LIMIT $k
                """,
                searchText=search_text,
                k=k,
            ).data()
        scores: dict[str, float] = {}
        for row in [*chunk_rows, *document_rows]:
            chunk_id = row.get("chunkId")
            if not chunk_id:
                continue
            scores[chunk_id] = max(scores.get(chunk_id, 0.0), float(row.get("score") or 0.0))
        return scores

    def hydrate_candidates(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        with self.session() as session:
            rows = session.run(
                """
                MATCH (chunk:Chunk)<-[:HAS_CHUNK]-(doc:Document)
                WHERE chunk.id IN $chunkIds
                OPTIONAL MATCH (doc)-[:HAS_TOPIC]->(topic:Topic)
                OPTIONAL MATCH (doc)-[rel:REFERENCES|REPEALS|AMENDS]->(target:Document)
                RETURN
                    chunk.id AS chunkId,
                    chunk.text AS text,
                    chunk.language AS language,
                    chunk.order AS chunkOrder,
                    chunk.headingPath AS headingPath,
                    doc.id AS documentId,
                    doc.sourceUrl AS sourceUrl,
                    doc.titleAr AS titleAr,
                    doc.titleEn AS titleEn,
                    doc.date AS date,
                    doc.type AS documentType,
                    doc.number AS number,
                    collect(DISTINCT coalesce(topic.canonicalName, topic.name)) AS topics,
                    collect(DISTINCT {
                        type: type(rel),
                        target: coalesce(target.titleEn, target.titleAr, target.id),
                        targetId: target.id,
                        context: rel.context
                    }) AS refs
                """,
                chunkIds=chunk_ids,
            ).data()
        hydrated: dict[str, dict[str, Any]] = {}
        for row in rows:
            refs = [ref for ref in row.get("refs", []) if ref.get("type") and ref.get("target")]
            topics = sorted(topic for topic in row.get("topics", []) if topic)
            title = row.get("titleEn") or row.get("titleAr") or row.get("documentId")
            ref_text = "; ".join(
                f"{ref['type']} {ref.get('target')}"
                for ref in refs[:5]
                if ref.get("type")
            )
            metadata = (
                f"Document: {title}\n"
                f"Date: {row.get('date')}\n"
                f"Type: {row.get('documentType')}\n"
                f"Number: {row.get('number')}\n"
                f"Topics: {', '.join(topics)}\n"
                f"References: {ref_text}"
            )
            chunk = {
                "id": row["chunkId"],
                "document_id": row["documentId"],
                "language": row.get("language"),
                "text": row.get("text") or "",
                "order": row.get("chunkOrder"),
                "heading_path": row.get("headingPath"),
            }
            hydrated[row["chunkId"]] = {
                "chunk": chunk,
                "document": {
                    "id": row["documentId"],
                    "title_en": row.get("titleEn"),
                    "title_ar": row.get("titleAr"),
                    "source_url": row.get("sourceUrl"),
                    "date": row.get("date"),
                    "document_type": row.get("documentType"),
                    "number": row.get("number"),
                },
                "expanded_text": f"{chunk['text']}\n\n--- Graph Context ---\n{metadata}",
            }
        return hydrated

    def search(self, query: str, *, k: int = 50, dense_weight: float = 0.35) -> list[dict[str, Any]]:
        vector_scores = normalize_score_map(self.vector_candidates(query, k=max(k, 50)))
        sparse_scores = normalize_score_map(self.fulltext_candidates(query, k=max(k, 50)))
        chunk_ids = sorted(set(vector_scores) | set(sparse_scores))
        hydrated = self.hydrate_candidates(chunk_ids)
        candidates: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            row = hydrated.get(chunk_id)
            if not row:
                continue
            dense_score = vector_scores.get(chunk_id, 0.0)
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            weighted_score = dense_weight * dense_score + (1.0 - dense_weight) * sparse_score
            score = max(sparse_score, weighted_score)
            candidates.append(
                {
                    **row,
                    "score": score,
                    "dense_score": dense_score,
                    "bm25_score": sparse_score,
                }
            )
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:k]

    def rerank(self, query: str, candidates: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
        return LocalHybridSearch.rerank(self, query, candidates, top_n=top_n)

    def synthesize(self, query: str, candidates: list[dict[str, Any]]) -> str:
        context_blocks = []
        for index, candidate in enumerate(candidates, start=1):
            document = candidate.get("document", {})
            title = document.get("title_en") or document.get("title_ar") or candidate["chunk"]["document_id"]
            source = document.get("source_url")
            context_blocks.append(f"[{index}] {title}\nURL: {source}\n{candidate['expanded_text'][:1800]}")
        context = "\n\n".join(context_blocks)
        answer = complete_text(
            "You answer legal search questions using only the supplied context. Cite bracket numbers.",
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer with a concise summary and citations.",
        )
        if answer:
            return answer
        if not candidates:
            return "No matching context was found."
        return "\n".join(
            [
                "LLM synthesis is disabled, so here is an extractive answer from Neo4j contexts:",
                *[
                    f"[{index}] {(candidate.get('document') or {}).get('title_en') or candidate['chunk']['document_id']}: "
                    f"{re.sub(r'\\s+', ' ', candidate['chunk'].get('text', '')).strip()[:350]}"
                    for index, candidate in enumerate(candidates, start=1)
                ],
            ]
        )


def render_matches(candidates: list[dict[str, Any]]) -> None:
    table = Table(title="Hybrid Matches")
    table.add_column("#", justify="right")
    table.add_column("Score")
    table.add_column("Dense")
    table.add_column("BM25")
    table.add_column("Rerank")
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
            f"{candidate['rerank_score']:.3f}" if candidate.get("rerank_score") is not None else "-",
            doc_id,
            topic_line[:60],
        )
    console.print(table)


def dedupe_by_document(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        doc_id = candidate["chunk"]["document_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(candidate)
    return unique


def create_searcher(
    backend: str,
    *,
    data_dir: Path = RAW_DIR,
    allow_self_signed: bool = False,
) -> Any:
    if backend == "neo4j":
        return Neo4jHybridSearch(allow_self_signed=allow_self_signed)
    if backend != "local":
        raise ValueError("backend must be 'local' or 'neo4j'")
    if not (data_dir / "documents.jsonl").exists() and (SAMPLE_DIR / "documents.jsonl").exists():
        data_dir = SAMPLE_DIR
    return LocalHybridSearch(data_dir)


def run_search_pipeline(
    searcher: Any,
    query_text: str,
    *,
    top_k: int = 50,
    top_n: int = 5,
    dense_weight: float = 0.35,
    rerank: bool = True,
) -> SearchPipelineResult:
    """Run the canonical hybrid -> graph expansion -> rerank -> synthesis flow."""
    if not query_text.strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or top_n < 1:
        raise ValueError("top_k and top_n must be positive")
    if top_n > top_k:
        raise ValueError("top_n must be less than or equal to top_k")
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("dense_weight must be between 0 and 1")

    candidates = searcher.search(query_text, k=top_k, dense_weight=dense_weight)
    if rerank:
        # Rerank the full candidate pool before document deduplication. Otherwise,
        # several chunks from one document can consume all final result slots.
        candidates = searcher.rerank(query_text, candidates, top_n=top_k)
    candidates = dedupe_by_document(candidates)[:top_n]
    return SearchPipelineResult(
        candidates=candidates,
        answer=searcher.synthesize(query_text, candidates),
        rerank_requested=rerank,
        rerank_applied=rerank and any(candidate.get("rerank_applied") is True for candidate in candidates),
    )


@app.command()
def query(
    query_text: str = typer.Option(..., "--query", "-q", help="User legal search query."),
    data_dir: Path = typer.Option(RAW_DIR, help="Directory containing documents/chunks/topics JSONL."),
    backend: str = typer.Option("local", help="Search backend: local or neo4j."),
    top_k: int = typer.Option(50, help="Candidate generation size."),
    top_n: int = typer.Option(5, help="Final contexts to synthesize."),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Run cross-encoder reranking (enabled by default)."),
    dense_weight: float = typer.Option(0.35, help="Hybrid score weight for dense similarity; remainder is BM25."),
    allow_self_signed: bool = typer.Option(False, help="Use neo4j+ssc when NEO4J_URI is neo4j+s and local cert validation fails."),
) -> None:
    try:
        searcher = create_searcher(backend, data_dir=data_dir, allow_self_signed=allow_self_signed)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        result = run_search_pipeline(
            searcher,
            query_text,
            top_k=top_k,
            top_n=top_n,
            dense_weight=dense_weight,
            rerank=rerank,
        )
        render_matches(result.candidates)
        console.rule("Graph Context + Final Answer")
        console.print(result.answer, markup=False, highlight=False)
    finally:
        close = getattr(searcher, "close", None)
        if close:
            close()


if __name__ == "__main__":
    app()
