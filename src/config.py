from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample_output"
STATE_DB = DATA_DIR / "crawl_state.sqlite3"


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    qanoon_api_url: str = "https://qanoon.om/wp-json/wp/v2/posts"
    decree_api_url: str = "https://decree.om/wp-json/wp/v2/posts"
    qanoon_base_url: str = "https://qanoon.om"
    decree_base_url: str = "https://decree.om"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password123")
    neo4j_database: str | None = os.getenv("NEO4J_DATABASE") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None
    google_model: str = os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    search_backend: str = os.getenv("SEARCH_BACKEND", "local").strip().lower()
    search_data_dir: Path = Path(os.getenv("SEARCH_DATA_DIR", str(SAMPLE_DIR)))
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "50"))
    search_top_n: int = int(os.getenv("SEARCH_TOP_N", "5"))
    search_dense_weight: float = float(os.getenv("SEARCH_DENSE_WEIGHT", "0.35"))
    search_rerank: bool = env_bool("SEARCH_RERANK", True)
    user_agent: str = (
        "Mozilla/5.0 (compatible; OmanLegalGraphRAG/1.0; "
        "+https://github.com/example/legal-graphrag-pipeline)"
    )
    crawler_proxy_url: str | None = os.getenv("CRAWLER_PROXY_URL") or None
    crawler_cookie_jar: str | None = os.getenv("CRAWLER_COOKIE_JAR") or None
    request_timeout: float = 30.0
    per_page: int = 100
    min_delay_seconds: float = 0.8
    max_delay_seconds: float = 2.4


settings = Settings()


def ensure_directories() -> None:
    for path in [DATA_DIR, RAW_DIR, SAMPLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
