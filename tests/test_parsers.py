from src.scraper.parsers import (
    classify_reference_context,
    extract_cross_references,
    extract_english_link,
    extract_number_from_detail_page,
    extract_pdf_urls,
    is_gazette_slug,
    parse_pdf_metadata,
)


def test_extract_english_link_follows_embedded_decree_url() -> None:
    html = '<p><a class="decree-link" href="https://decree.om/2026/rd20260057/">English</a></p>'
    assert extract_english_link(html) == "https://decree.om/2026/rd20260057/"


def test_extract_english_link_accepts_decree_hostname_without_label() -> None:
    html = '<p><a href="https://decree.om/2026/rd20260057/">ترجمة</a></p>'
    assert extract_english_link(html) == "https://decree.om/2026/rd20260057/"


def test_gazette_slug_detection() -> None:
    assert is_gazette_slug("og1649")
    assert not is_gazette_slug("rd2026060")


def test_pdf_metadata_extracts_ministerial_issuer() -> None:
    url = "https://data.qanoon.om/ar/md/mjla/2026-0050.pdf"
    assert parse_pdf_metadata(url) == {
        "family": "md",
        "year": "2026",
        "issuer_code": "mjla",
        "serial": "50",
    }


def test_pdf_metadata_extracts_royal_decree_family() -> None:
    url = "https://data.qanoon.om/ar/rd/2026/2026-060.pdf"
    assert parse_pdf_metadata(url)["family"] == "rd"
    assert parse_pdf_metadata(url)["serial"] == "60"


def test_number_parser_targets_paragraph_below_entry_title_only() -> None:
    detail_html = """
    <article>
      <h1 class="entry-title">المرسوم السلطاني ٦٠ / ٢٠٢٦ بإلغاء المعهد</h1>
      <div class="intro-text section-inner max-percentage small">
        <p>2026/60 60/2026  ٢٠٢٦/٦٠ ٦٠/٢٠٢٦</p>
      </div>
      <div class="entry-content">
        <p>مرسوم آخر رقم 31/2006 داخل المحتوى لا يجب أن يكون الرقم الأساسي.</p>
      </div>
    </article>
    """
    assert extract_number_from_detail_page(detail_html) == "60/2026"


def test_number_parser_does_not_scan_body_when_exact_block_missing() -> None:
    detail_html = """
    <article>
      <h1 class="entry-title">title</h1>
      <div class="entry-content"><p>2026/99 99/2026</p></div>
    </article>
    """
    assert extract_number_from_detail_page(detail_html) is None


def test_cross_reference_keywords_classify_relationships() -> None:
    html = """
    <p>بعد الاطلاع على <a href="https://qanoon.om/p/2021/rd2021006/">النظام الأساسي</a>،</p>
    <p>يلغى <a href="https://qanoon.om/p/2006/rd2006031/">النظام القديم</a>.</p>
    <p>تعدل بعض أحكام <a href="https://qanoon.om/p/2012/rd2012071/">القانون</a>.</p>
    <p>يستبدل نص المادة في <a href="https://qanoon.om/p/2013/rd2013033/">مرسوم</a>.</p>
    <p>يضاف بند إلى <a href="https://qanoon.om/p/2020/rd2020075/">نظام</a>.</p>
    """
    refs = extract_cross_references(html, "rd2026060")
    rels = {ref.target_slug: ref.rel_type for ref in refs}
    assert rels["rd2021006"] == "REFERENCES"
    assert rels["rd2006031"] == "REPEALS"
    assert rels["rd2012071"] == "AMENDS"
    assert rels["rd2013033"] == "AMENDS"
    assert rels["rd2020075"] == "AMENDS"


def test_keyword_classifier_has_expanded_amend_forms() -> None:
    for phrase in ["يعدل القانون", "تعدل اللائحة", "يستبدل النص", "يضاف بند"]:
        assert classify_reference_context(phrase) == "AMENDS"


def test_extract_pdf_urls_keeps_data_qanoon_links() -> None:
    html = '<a class="pdf-link" href="https://data.qanoon.om/ar/rd/2026/2026-057.pdf">تحميل</a>'
    assert extract_pdf_urls(html) == ["https://data.qanoon.om/ar/rd/2026/2026-057.pdf"]
