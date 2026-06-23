from __future__ import annotations

from tenacity.wait import wait_exponential_jitter

from src.scraper.crawl import QanoonCrawler, USER_AGENTS


class StubResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        pass


class RecordingClient:
    def __init__(self) -> None:
        self.headers: dict[str, str] | None = None

    def get(self, url: str, *, headers: dict[str, str], **kwargs) -> StubResponse:
        self.headers = headers
        return StubResponse()


def test_get_rotates_user_agent(monkeypatch) -> None:
    crawler = object.__new__(QanoonCrawler)
    crawler.headers = {"User-Agent": USER_AGENTS[0], "Accept": "application/json"}
    crawler.client = RecordingClient()
    crawler.pace = lambda: None
    monkeypatch.setattr("src.scraper.crawl.random.choice", lambda choices: choices[-1])
    crawler.get("https://example.test")
    assert crawler.client.headers is not None
    assert crawler.client.headers["User-Agent"] == USER_AGENTS[-1]


def test_get_uses_jittered_exponential_backoff() -> None:
    assert isinstance(QanoonCrawler.get.retry.wait, wait_exponential_jitter)
    assert QanoonCrawler.get.retry.stop.max_attempt_number == 4
