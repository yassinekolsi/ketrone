from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


class CrawlState:
    """Small SQLite-backed crawl checkpoint store."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS discovered_posts (
                slug TEXT PRIMARY KEY,
                post_id INTEGER,
                source_url TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                retries INTEGER NOT NULL DEFAULT 0,
                is_gazette INTEGER NOT NULL DEFAULT 0,
                english_url TEXT,
                error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS crawl_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def upsert_discovered(
        self,
        slug: str,
        post_id: int | None,
        source_url: str,
        *,
        is_gazette: bool = False,
        english_url: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO discovered_posts(slug, post_id, source_url, is_gazette, english_url)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                post_id=COALESCE(excluded.post_id, discovered_posts.post_id),
                source_url=excluded.source_url,
                is_gazette=excluded.is_gazette,
                english_url=COALESCE(excluded.english_url, discovered_posts.english_url),
                updated_at=CURRENT_TIMESTAMP
            """,
            (slug, post_id, source_url, int(is_gazette), english_url),
        )
        self.conn.commit()

    def mark_status(self, slug: str, status: str, error: str | None = None) -> None:
        self.conn.execute(
            """
            UPDATE discovered_posts
            SET status=?, error=?, retries=CASE WHEN ? IS NULL THEN retries ELSE retries + 1 END,
                updated_at=CURRENT_TIMESTAMP
            WHERE slug=?
            """,
            (status, error, error, slug),
        )
        self.conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO crawl_meta(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """,
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM crawl_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def pending_slugs(self) -> Iterable[str]:
        return [row["slug"] for row in self.pending_records()]

    def pending_records(self, *, max_retries: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT slug, post_id, source_url, status, retries, is_gazette, english_url, error
            FROM discovered_posts
            WHERE status IN ('pending', 'failed') AND retries <= ?
            ORDER BY updated_at
            """,
            (max_retries,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()
