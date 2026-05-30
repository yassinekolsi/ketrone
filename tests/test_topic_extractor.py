from src.llm_agents.topic_extractor import fallback_topics


def test_fallback_topics_detects_transport_from_postal_terms() -> None:
    topics = fallback_topics(
        {
            "id": "rd2026057",
            "title_en": "Royal Decree regarding the Postal Sector",
            "content_en": "This decree regulates postal services.",
        }
    )
    assert any(topic["canonical_name"] == "Transport" for topic in topics)
