from src.scraper.state import CrawlState


def test_pending_records_include_failed_and_pending_for_resume(tmp_path) -> None:
    state = CrawlState(tmp_path / "crawl.sqlite3")
    try:
        state.upsert_discovered("rd2026060", 60, "https://qanoon.om/p/2026/rd2026060/")
        state.upsert_discovered("rd2026059", 59, "https://qanoon.om/p/2026/rd2026059/")
        state.mark_status("rd2026059", "failed", "timeout")

        pending = state.pending_records()
        assert {row["slug"] for row in pending} == {"rd2026060", "rd2026059"}
        assert set(state.pending_slugs()) == {"rd2026060", "rd2026059"}
    finally:
        state.close()
