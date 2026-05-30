from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from neo4j import GraphDatabase
from rich.console import Console

from src.config import RAW_DIR, settings
from src.io import read_jsonl


app = typer.Typer(help="Load scraped documents, topics, chunks, and vectors into Neo4j.")
console = Console()


class GraphLoader:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def setup_schema(self, *, vector_dim: int | None = None) -> None:
        statements = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT topic_canonical IF NOT EXISTS FOR (t:Topic) REQUIRE t.canonicalName IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX document_slug IF NOT EXISTS FOR (d:Document) ON (d.slug)",
            "CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE FULLTEXT INDEX document_content_ft IF NOT EXISTS FOR (d:Document) ON EACH [d.titleAr, d.titleEn, d.contentAr, d.contentEn]",
            "CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
        ]
        if vector_dim:
            statements.extend(
                [
                    (
                        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
                        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {vector_dim}, `vector.similarity_function`: 'cosine'}}}}"
                    ),
                    (
                        "CREATE VECTOR INDEX topic_embedding IF NOT EXISTS FOR (t:Topic) ON (t.embedding) "
                        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {vector_dim}, `vector.similarity_function`: 'cosine'}}}}"
                    ),
                ]
            )
        with self.session() as session:
            for statement in statements:
                session.run(statement)

    def load_documents(self, documents: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (d:Document {id: row.id})
        SET d.slug = row.slug,
            d.sourceUrl = row.source_url,
            d.titleAr = row.title_ar,
            d.titleEn = row.title_en,
            d.date = row.date,
            d.type = row.document_type,
            d.number = row.number,
            d.issuerCode = row.issuer_code,
            d.pdfUrlAr = row.pdf_url_ar,
            d.pdfUrlEn = row.pdf_url_en,
            d.englishUrl = row.english_url,
            d.contentAr = row.content_ar,
            d.contentEn = row.content_en
        """
        with self.session() as session:
            session.run(query, rows=documents)
            for document in documents:
                for ref in document.get("cross_references", []):
                    target_slug = ref.get("target_slug")
                    rel_type = ref.get("rel_type", "REFERENCES")
                    if not target_slug or rel_type not in {"REFERENCES", "REPEALS", "AMENDS"}:
                        continue
                    session.run(
                        f"""
                        MATCH (a:Document {{id: $source}})
                        MERGE (b:Document {{id: $target}})
                        ON CREATE SET b.slug = $target
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r.context = $context, r.anchorText = $anchorText, r.targetUrl = $targetUrl
                        """,
                        source=document["id"],
                        target=target_slug,
                        context=ref.get("context"),
                        anchorText=ref.get("anchor_text"),
                        targetUrl=ref.get("target_url"),
                    )

    def load_chunks(self, chunks: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (c:Chunk {id: row.id})
        SET c.text = row.text,
            c.language = row.language,
            c.order = row.order,
            c.headingPath = row.heading_path,
            c.embedding = row.embedding
        WITH c, row
        MATCH (d:Document {id: row.document_id})
        MERGE (d)-[r:HAS_CHUNK {language: row.language}]->(c)
        SET r.order = row.order
        """
        with self.session() as session:
            session.run(query, rows=chunks)

    def load_topics(self, topic_links: list[dict[str, Any]]) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (t:Topic {canonicalName: row.canonical_name})
        SET t.name = row.name,
            t.embedding = row.embedding
        WITH t, row
        MATCH (d:Document {id: row.document_id})
        MERGE (d)-[r:HAS_TOPIC]->(t)
        SET r.confidence = row.confidence,
            r.evidence = row.evidence
        """
        with self.session() as session:
            session.run(query, rows=topic_links)


def infer_vector_dim(*collections: list[dict[str, Any]]) -> int | None:
    for rows in collections:
        for row in rows:
            embedding = row.get("embedding")
            if isinstance(embedding, list) and embedding:
                return len(embedding)
    return None


@app.command()
def main(
    documents_path: Path = typer.Option(RAW_DIR / "documents.jsonl"),
    chunks_path: Path = typer.Option(RAW_DIR / "chunks.embedded.jsonl"),
    topics_path: Path = typer.Option(RAW_DIR / "topics.embedded.jsonl"),
    setup_only: bool = typer.Option(False, help="Only create constraints/indexes."),
) -> None:
    documents = read_jsonl(documents_path)
    chunks = read_jsonl(chunks_path)
    topics = read_jsonl(topics_path)
    vector_dim = infer_vector_dim(chunks, topics)
    loader = GraphLoader(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database)
    try:
        loader.setup_schema(vector_dim=vector_dim)
        if setup_only:
            console.print("[green]schema ready[/]")
            return
        if documents:
            loader.load_documents(documents)
        if chunks:
            loader.load_chunks(chunks)
        if topics:
            loader.load_topics(topics)
    finally:
        loader.close()
    console.print(f"[green]loaded {len(documents)} documents, {len(chunks)} chunks, {len(topics)} topic links[/]")


if __name__ == "__main__":
    app()
