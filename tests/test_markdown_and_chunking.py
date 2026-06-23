from src.llm_agents.chunking import chunk_document
from src.scraper.markdown import html_to_markdown


def test_markdown_conversion_strips_scripts_and_keeps_headings() -> None:
    html = "<script>alert(1)</script><h2>Article 1</h2><p><strong>Text</strong></p>"
    markdown = html_to_markdown(html)
    assert "alert" not in markdown
    assert "## Article 1" in markdown
    assert "**Text**" in markdown


def test_markdown_conversion_strips_layout_chrome_and_keeps_tables() -> None:
    html = """
    <header>Site navigation</header>
    <article>
      <table><tr><th>Ministry</th><th>Decision</th></tr><tr><td>Health</td><td>60/2026</td></tr></table>
    </article>
    <footer>Copyright</footer>
    """
    markdown = html_to_markdown(html)
    assert "Site navigation" not in markdown
    assert "Copyright" not in markdown
    assert "| Ministry | Decision |" in markdown
    assert "| Health | 60/2026 |" in markdown


def test_chunk_document_uses_language_specific_content() -> None:
    document = {
        "id": "rd2026060",
        "content_en": "# Royal Decree\n\n## Article 1\n\nAbolishes the institute.",
    }
    chunks = chunk_document(document, language="en", max_tokens=20, overlap_tokens=2)
    assert len(chunks) == 1
    assert chunks[0].document_id == "rd2026060"
    assert chunks[0].language == "en"
    assert "Abolishes" in chunks[0].text
