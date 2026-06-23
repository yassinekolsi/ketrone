from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from src.config import settings
from src.search_client import create_searcher, run_search_pipeline


logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=settings.search_top_k, ge=1, le=100)
    top_n: int = Field(default=settings.search_top_n, ge=1, le=20)
    dense_weight: float = Field(default=settings.search_dense_weight, ge=0.0, le=1.0)
    rerank: bool = settings.search_rerank

    @model_validator(mode="after")
    def validate_depths(self) -> "SearchRequest":
        if self.top_n > self.top_k:
            raise ValueError("top_n must be less than or equal to top_k")
        if not self.query.strip():
            raise ValueError("query must not be blank")
        return self


class SearchSource(BaseModel):
    rank: int
    document_id: str
    chunk_id: str | None = None
    title: str | None = None
    source_url: str | None = None
    language: str | None = None
    snippet: str
    hybrid_score: float
    dense_score: float
    lexical_score: float
    rerank_score: float | None = None


class PipelineMetadata(BaseModel):
    backend: str
    candidate_generation: str = "dense+BM25"
    graph_expansion: bool = True
    rerank_requested: bool
    rerank_applied: bool
    reranker_model: str | None = None


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchSource]
    pipeline: PipelineMetadata


def build_searcher() -> Any:
    return create_searcher(settings.search_backend, data_dir=settings.search_data_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.searcher = build_searcher()
    try:
        yield
    finally:
        close = getattr(app.state.searcher, "close", None)
        if close:
            close()


app = FastAPI(
    title="Oman Legal GraphRAG API",
    summary="Three-stage hybrid legal retrieval over Omani legislation.",
    version="1.0.0",
    lifespan=lifespan,
)


def searcher_from(request: Request) -> Any:
    searcher = getattr(request.app.state, "searcher", None)
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search backend is not initialized")
    return searcher


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {
        "service": "Oman Legal GraphRAG API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    """Cheap liveness endpoint for Render and UptimeRobot."""
    return {"status": "ok"}


@app.get("/ready", tags=["service"])
def ready(request: Request) -> dict[str, str]:
    """Check that the configured retrieval backend is usable."""
    searcher = searcher_from(request)
    try:
        verify = getattr(searcher, "verify_connectivity", None)
        if verify:
            verify()
        elif not getattr(searcher, "chunks", None):
            raise RuntimeError("No local chunks are loaded")
    except Exception as exc:
        logger.warning("Search readiness check failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Search backend is unavailable") from exc
    return {"status": "ready", "backend": settings.search_backend}


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    searcher = searcher_from(request)
    try:
        result = run_search_pipeline(
            searcher,
            payload.query,
            top_k=payload.top_k,
            top_n=payload.top_n,
            dense_weight=payload.dense_weight,
            rerank=payload.rerank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search request failed")
        raise HTTPException(status_code=503, detail="Search backend is temporarily unavailable") from exc

    sources: list[SearchSource] = []
    for rank, candidate in enumerate(result.candidates, start=1):
        chunk = candidate["chunk"]
        document = candidate.get("document") or {}
        title = (
            document.get("title_en")
            or document.get("title_ar")
            or document.get("titleEn")
            or document.get("titleAr")
        )
        source_url = document.get("source_url") or document.get("sourceUrl")
        snippet = " ".join(str(chunk.get("text") or "").split())[:600]
        sources.append(
            SearchSource(
                rank=rank,
                document_id=chunk["document_id"],
                chunk_id=chunk.get("id"),
                title=title,
                source_url=source_url,
                language=chunk.get("language"),
                snippet=snippet,
                hybrid_score=float(candidate.get("score") or 0.0),
                dense_score=float(candidate.get("dense_score") or 0.0),
                lexical_score=float(candidate.get("bm25_score") or 0.0),
                rerank_score=candidate.get("rerank_score"),
            )
        )

    return SearchResponse(
        query=payload.query,
        answer=result.answer,
        sources=sources,
        pipeline=PipelineMetadata(
            backend=settings.search_backend,
            rerank_requested=result.rerank_requested,
            rerank_applied=result.rerank_applied,
            reranker_model=settings.reranker_model if result.rerank_requested else None,
        ),
    )
