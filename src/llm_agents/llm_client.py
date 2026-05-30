from __future__ import annotations

import json
from typing import Any

import httpx
from openai import OpenAI

from src.config import settings


def openai_available() -> bool:
    return bool(settings.openai_api_key)


def gemini_available() -> bool:
    return bool(settings.google_api_key)


def _gemini_generate(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    json_mode: bool = False,
) -> str | None:
    if not settings.google_api_key:
        return None
    model_name = model or settings.google_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.0 if json_mode else 0.1},
    }
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    response = httpx.post(
        url,
        params={"key": settings.google_api_key},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip() or None


def complete_json(system_prompt: str, user_prompt: str, *, model: str | None = None) -> dict[str, Any] | None:
    if settings.google_api_key:
        try:
            content = _gemini_generate(system_prompt, user_prompt, model=model, json_mode=True)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError):
            return None
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    if not settings.openai_api_key:
        return None
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=model or settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def complete_text(system_prompt: str, user_prompt: str, *, model: str | None = None) -> str | None:
    if settings.google_api_key:
        try:
            return _gemini_generate(system_prompt, user_prompt, model=model, json_mode=False)
        except httpx.HTTPError:
            return None
    if not settings.openai_api_key:
        return None
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=model or settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
