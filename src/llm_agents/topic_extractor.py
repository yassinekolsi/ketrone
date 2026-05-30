from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.console import Console

from src.config import RAW_DIR
from src.io import read_jsonl, write_jsonl
from src.llm_agents.llm_client import complete_json


app = typer.Typer(help="Extract legal topics from consolidated documents.")
console = Console()

SEED_TAXONOMY = [
    "Administrative Law",
    "Banking and Finance",
    "Civil Service",
    "Commerce and Companies",
    "Criminal Law",
    "Customs",
    "Education",
    "Employment and Labour",
    "Environment",
    "Healthcare",
    "Insurance",
    "International Treaties",
    "Investment",
    "Judiciary",
    "Maritime Law",
    "Municipal Affairs",
    "Oil and Gas",
    "Public Procurement",
    "Real Estate",
    "Sports",
    "Taxation",
    "Telecommunications",
    "Transport",
    "Urban Planning",
]

FALLBACK_KEYWORDS = {
    "Taxation": ["ضريبة", "tax", "vat"],
    "Employment and Labour": ["عمل", "عامل", "labour", "labor", "worker", "omanization"],
    "Healthcare": ["صحي", "الصحة", "medical", "health"],
    "Education": ["تعليم", "جامعة", "school", "education", "university"],
    "Banking and Finance": ["بنك", "مصرف", "finance", "bank", "cbo"],
    "Urban Planning": ["التخطيط العمراني", "urban planning"],
    "Transport": ["نقل", "transport", "postal", "البريد"],
    "Judiciary": ["قضائي", "محكمة", "judicial", "court"],
    "International Treaties": ["اتفاقية", "treaty", "agreement"],
    "Commerce and Companies": ["تجارة", "شركة", "commercial", "company"],
    "Environment": ["بيئة", "environment"],
    "Sports": ["رياض", "sports"],
    "Public Procurement": ["مناقصة", "tender", "procurement"],
}


def normalize_topic_name(name: str) -> str:
    compact = re.sub(r"\s+", " ", name.strip())
    for seed in SEED_TAXONOMY:
        if compact.casefold() == seed.casefold():
            return seed
    return compact.title()


def fallback_topics(document: dict) -> list[dict]:
    text = " ".join(
        str(document.get(key) or "")
        for key in ["title_ar", "title_en", "content_ar", "content_en", "titleAr", "titleEn", "contentAr", "contentEn"]
    ).lower()
    topics: list[dict] = []
    for topic, keywords in FALLBACK_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            topics.append({"name": topic, "canonical_name": topic, "confidence": 0.62, "evidence": "keyword fallback"})
    if not topics:
        doc_type = document.get("document_type") or document.get("documentType")
        if doc_type:
            topics.append({"name": normalize_topic_name(str(doc_type).replace("_", " ")), "canonical_name": normalize_topic_name(str(doc_type).replace("_", " ")), "confidence": 0.45, "evidence": "document type fallback"})
    return topics[:6]


def llm_topics(document: dict) -> list[dict] | None:
    content = document.get("content_en") or document.get("contentEn") or document.get("content_ar") or document.get("contentAr") or ""
    content = content[:12000]
    system = (
        "You extract legal topics from Omani legislation. Return JSON only. "
        "Prefer the seed taxonomy when it fits; add a new topic only if necessary."
    )
    user = f"""
Seed taxonomy:
{SEED_TAXONOMY}

Document title:
{document.get('title_en') or document.get('title_ar') or document.get('titleEn') or document.get('titleAr')}

Document content:
{content}

Return this shape:
{{"topics":[{{"name":"...", "canonical_name":"...", "confidence":0.0, "evidence":"short phrase"}}]}}
"""
    payload = complete_json(system, user)
    if not payload:
        return None
    topics = payload.get("topics")
    if not isinstance(topics, list):
        return None
    cleaned = []
    for topic in topics:
        if not isinstance(topic, dict) or not topic.get("name"):
            continue
        name = normalize_topic_name(str(topic["name"]))
        canonical = normalize_topic_name(str(topic.get("canonical_name") or name))
        cleaned.append(
            {
                "name": name,
                "canonical_name": canonical,
                "confidence": float(topic.get("confidence") or 0.0),
                "evidence": str(topic.get("evidence") or "")[:300],
            }
        )
    return cleaned[:8]


def extract_topics_for_documents(documents: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for document in documents:
        topics = llm_topics(document) or fallback_topics(document)
        for topic in topics:
            rows.append({"document_id": document["id"], **topic})
    return rows


@app.command()
def main(
    input_path: Path = typer.Option(RAW_DIR / "documents.jsonl"),
    output_path: Path = typer.Option(RAW_DIR / "topics.jsonl"),
) -> None:
    documents = read_jsonl(input_path)
    topics = extract_topics_for_documents(documents)
    write_jsonl(output_path, topics)
    console.print(f"[green]wrote {len(topics)} topic links to {output_path}[/]")


if __name__ == "__main__":
    app()
