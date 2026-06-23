from __future__ import annotations

from src.search_client import run_search_pipeline


def candidate(document_id: str, chunk_id: str) -> dict:
    return {
        "chunk": {"id": chunk_id, "document_id": document_id, "text": chunk_id},
        "document": {"id": document_id},
        "expanded_text": f"{chunk_id}\nGraph Context",
        "score": 0.5,
        "dense_score": 0.4,
        "bm25_score": 0.6,
    }


class FakeSearcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, *, k: int, dense_weight: float) -> list[dict]:
        self.calls.append("search")
        return [candidate("doc-a", "a-1"), candidate("doc-a", "a-2"), candidate("doc-b", "b-1")]

    def rerank(self, query: str, candidates: list[dict], *, top_n: int) -> list[dict]:
        self.calls.append("rerank")
        ranked = [candidates[2], candidates[1], candidates[0]]
        for item in ranked:
            item["rerank_applied"] = True
            item["rerank_score"] = 1.0
        return ranked[:top_n]

    def synthesize(self, query: str, candidates: list[dict]) -> str:
        self.calls.append("synthesize")
        return ",".join(item["chunk"]["document_id"] for item in candidates)


def test_default_pipeline_reranks_before_document_deduplication() -> None:
    searcher = FakeSearcher()
    result = run_search_pipeline(searcher, "legal question", top_k=3, top_n=2)
    assert searcher.calls == ["search", "rerank", "synthesize"]
    assert [item["chunk"]["document_id"] for item in result.candidates] == ["doc-b", "doc-a"]
    assert result.rerank_requested is True
    assert result.rerank_applied is True


def test_pipeline_can_explicitly_skip_reranking() -> None:
    searcher = FakeSearcher()
    result = run_search_pipeline(searcher, "legal question", top_k=3, top_n=2, rerank=False)
    assert searcher.calls == ["search", "synthesize"]
    assert result.rerank_requested is False
    assert result.rerank_applied is False
