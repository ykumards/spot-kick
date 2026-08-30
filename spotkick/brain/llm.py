"""The brain backend protocol and its factory.

A backend implements ``complete_json(prompt, schema) -> dict``: structured output validated against ``schema``.
Two backends exist, the Codex CLI and the Claude Code CLI; ``cfg.llm_backend`` selects one. The brain proposes
song names only; track ids come from ``brain.resolve``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..config import Config

if TYPE_CHECKING:
    from .claude import ClaudeCode
    from .codex import Codex

BACKEND_NAMES = ("codex", "claude")

class Backend(Protocol):
    name: str

    def complete_json(self, prompt: str, schema: dict, *, timeout: int = 240) -> dict: ...


class BrainError(RuntimeError):
    pass


def make_backend(cfg: Config) -> Codex | ClaudeCode:
    # Deferred: the backend modules import BrainError from this module, so top-level imports would be circular.
    from .claude import ClaudeCode
    from .codex import Codex

    if cfg.llm_backend == "claude":
        return ClaudeCode(model=cfg.claude_model)
    if cfg.llm_backend != "codex":
        raise BrainError(f"unknown llm_backend {cfg.llm_backend!r}; choose one of {', '.join(BACKEND_NAMES)}")
    return Codex(model=cfg.llm_model, reasoning=cfg.llm_reasoning)

