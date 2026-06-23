from __future__ import annotations

from fastapi.testclient import TestClient

from src import api


class FakeApiSearcher:
    def search(self, query: str, *, k: int, dense_weight: float) -> list[dict]:
        return [
            {
                "chunk": {
                    "id": "rd2026060:en:0",
                    "document_id": "rd2026060",
                    "language": "en",
                    "text": "The Higher Institute of Health Specialties is abolished.",
                },
                "document": {
                    "title_en": "Royal Decree 60/2026",
                    "source_url": "https://decree.om/2026/rd2026060/",
                },
                "expanded_text": "text\nGraph Context",
                "score": 0.9,
                "dense_score": 0.8,
                "bm25_score": 1.0,
            }
        ]

    def rerank(self, query: str, candidates: list[dict], *, top_n: int) -> list[dict]:
        candidates[0]["rerank_applied"] = True
        candidates[0]["rerank_score"] = 2.5
        return candidates

    def synthesize(self, query: str, candidates: list[dict]) -> str:
        return "Royal Decree 60/2026 abolished the institute [1]."

    def close(self) -> None:
        pass


def test_health_docs_and_search_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "build_searcher", FakeApiSearcher)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        response = client.post("/search", json={"query": "What abolished the institute?", "top_k": 5, "top_n": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["document_id"] == "rd2026060"
    assert body["pipeline"]["graph_expansion"] is True
    assert body["pipeline"]["rerank_applied"] is True
