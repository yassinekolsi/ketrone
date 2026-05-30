from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CrossReference:
    source_slug: str
    target_url: str
    target_slug: str | None
    rel_type: str
    anchor_text: str
    context: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LegalDocument:
    id: str
    slug: str
    source_url: str
    title_ar: str | None = None
    title_en: str | None = None
    date: str | None = None
    document_type: str | None = None
    number: str | None = None
    issuer_code: str | None = None
    pdf_url_ar: str | None = None
    pdf_url_en: str | None = None
    english_url: str | None = None
    content_ar: str | None = None
    content_en: str | None = None
    categories: list[int] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cross_references"] = [ref.to_dict() for ref in self.cross_references]
        return data


@dataclass
class Chunk:
    id: str
    document_id: str
    language: str
    text: str
    order: int
    heading_path: str | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Topic:
    name: str
    canonical_name: str
    confidence: float = 0.0
    evidence: str | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
