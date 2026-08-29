"""The Brain's contract. Not an agent: a stateless candidate generator with two operations.

    complete_json(prompt, schema) -> dict     structured output, validated against `schema` by the backend
    search_uri(artist, title) -> str | None   a Spotify track URI found with live search (optional per backend)

Backends: `codex` (the Codex CLI, the author's ChatGPT login), `openai` (the OpenAI API or any OpenAI-compatible
server — llama.cpp's `llama-server`, Ollama — chosen by base_url). Claude is deliberately not an option.
"""
from __future__ import annotations

from typing import Protocol


class Backend(Protocol):
    name: str

    def complete_json(self, prompt: str, schema: dict, *, timeout: int = 240) -> dict: ...

    def search_uri(self, artist: str, title: str) -> str | None: ...


class BrainError(RuntimeError):
    pass


def make_backend(cfg) -> Backend:
    if cfg.llm_backend == "codex":
        from .codex import Codex
        return Codex(model=cfg.llm_model, reasoning=cfg.llm_reasoning)
    if cfg.llm_backend in ("openai", "local"):
        from .openai_compat import OpenAICompat
        base_url = cfg.local_base_url if cfg.llm_backend == "local" else None
        return OpenAICompat(model=cfg.llm_model, base_url=base_url, reasoning=cfg.llm_reasoning)
    raise BrainError(f"unknown llm backend {cfg.llm_backend!r}")
