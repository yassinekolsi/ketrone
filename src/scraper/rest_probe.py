from __future__ import annotations

import httpx

from src.config import settings


def endpoint_returns_posts(url: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Return whether a WordPress posts endpoint returns JSON objects with ids."""
    try:
        response = httpx.get(url, params={"per_page": 1}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "id" in payload[0]:
        return True, f"ok id={payload[0]['id']}"
    return False, "response was not a non-empty post list with an id field"


def probe_required_endpoints() -> dict[str, tuple[bool, str]]:
    return {
        "qanoon": endpoint_returns_posts(settings.qanoon_api_url, settings.request_timeout),
        "decree": endpoint_returns_posts(settings.decree_api_url, settings.request_timeout),
    }


if __name__ == "__main__":
    for name, (ok, detail) in probe_required_endpoints().items():
        print(f"{name}: {'REST' if ok else 'fallback'} - {detail}")
