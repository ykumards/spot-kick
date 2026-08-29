"""OpenAI-compatible backend over plain HTTP: the OpenAI API (api.openai.com) or a local server that speaks the
same chat-completions shape — llama.cpp's `llama-server` (grammar-enforced JSON), Ollama, LM Studio.

No SDK; one POST. Structured output via `response_format: json_schema`, which all of the above support.
"""
from __future__ import annotations

import json
import os
import re

import requests

from .llm import BrainError

OPENAI_URL = "https://api.openai.com/v1"
URI_RE = re.compile(r"spotify:track:[A-Za-z0-9]{22}")


class OpenAICompat:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, reasoning: str | None = "low",
                 session: requests.Session | None = None):
        self.base_url = (base_url or OPENAI_URL).rstrip("/")
        self.local = base_url is not None
        self.name = "local" if self.local else "openai"
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ("local" if self.local else None)
        self.reasoning = reasoning
        self.http = session or requests.Session()

    def _post(self, path: str, body: dict, timeout: int) -> dict:
        if not self.api_key:
            raise BrainError("OPENAI_API_KEY is not set")
        r = self.http.post(f"{self.base_url}{path}", json=body, timeout=timeout,
                           headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        if r.status_code >= 400:
            raise BrainError(f"{self.name} {r.status_code}: {r.text[:300]}")
        return r.json()

    def complete_json(self, prompt: str, schema: dict, *, timeout: int = 240) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        }
        if self.reasoning and not self.local:
            body["reasoning_effort"] = self.reasoning
        data = self._post("/chat/completions", body, timeout)
        try:
            text = data["choices"][0]["message"]["content"]
            return json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            raise BrainError(f"{self.name} returned no JSON: {str(data)[:200]}") from e

    def search_uri(self, artist: str, title: str) -> str | None:
        """No live search here. A Spotify Web API searcher is the exact path (see brain/spotify_api.py, later)."""
        return None
