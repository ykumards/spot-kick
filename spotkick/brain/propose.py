"""Candidate proposals from the brain, deduplicated against the store.

The output is names only; resolving, embedding and choosing happen in ``kick.session``.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .prompts import CANDIDATES_SCHEMA, Context, candidates_prompt

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
BARE_URL_RE = re.compile(r"https?://\S+")

DEFAULT_REACH = "adjacent"
REJECT_DUPLICATE = "duplicate in set"
REJECT_KNOWN = "already known"

Logger = Callable[[str], None]


def ignore_log(message: str) -> None:
    return None


@dataclass
class Candidate:
    reach: str
    direction: str
    artist: str
    title: str
    why: str
    rejected_reason: str | None = None

    def as_row(self) -> dict:
        return {
            "reach": self.reach,
            "direction": self.direction,
            "artist": self.artist,
            "title": self.title,
            "why": self.why,
            "rejected_reason": self.rejected_reason,
        }

    def dedup_key(self) -> str:
        return f"{self.artist.lower()}|{self.title.lower()}"


def strip_links(text: str) -> str:
    """Remove markdown links and bare URLs from free text."""
    without_markdown = MARKDOWN_LINK_RE.sub(r"\1", text or "")
    return BARE_URL_RE.sub("", without_markdown).strip()


def parse_candidates(raw_candidates: list[dict]) -> list[Candidate]:
    """Convert the brain's rows to Candidates, dropping rows without an artist or title."""
    candidates = []
    for raw in raw_candidates:
        artist = (raw.get("artist") or "").strip()
        title = (raw.get("title") or "").strip()
        if not artist or not title:
            continue
        candidate = Candidate(
            reach=raw.get("reach") or DEFAULT_REACH,
            direction=strip_links(raw.get("direction", "")),
            artist=artist,
            title=title,
            why=strip_links(raw.get("why", "")),
        )
        candidates.append(candidate)
    return candidates


def mark_rejects(candidates: list[Candidate], store, seen_keys: set[str]) -> None:
    """Mark duplicates within the set and candidates the store has already seen."""
    for candidate in candidates:
        key = candidate.dedup_key()
        if key in seen_keys:
            candidate.rejected_reason = REJECT_DUPLICATE
        elif store.seen(candidate.artist, candidate.title):
            candidate.rejected_reason = REJECT_KNOWN
        seen_keys.add(key)


def ask_brain(backend, prompt: str, timeout: int) -> list[Candidate]:
    response = backend.complete_json(prompt, CANDIDATES_SCHEMA, timeout=timeout)
    return parse_candidates(response.get("candidates", []))


def propose(
    backend,
    store,
    *,
    n: int = 6,
    reach: str | None = None,
    lean: str | None = None,
    misses: list[dict] | None = None,
    min_fresh: int = 3,
    timeout: int = 240,
    log: Logger = ignore_log,
) -> list[Candidate]:
    """Ask the brain for candidates, mark the rejects, and ask once more if too few are fresh."""
    context = Context.from_store(store)
    first_prompt = candidates_prompt(context, n=n, reach=reach, lean=lean, misses=misses)
    candidates = ask_brain(backend, first_prompt, timeout)
    seen_keys: set[str] = set()
    mark_rejects(candidates, store, seen_keys)
    fresh = [candidate for candidate in candidates if not candidate.rejected_reason]
    if len(fresh) >= min_fresh:
        return candidates
    rejects = [f"{candidate.artist} — {candidate.title}" for candidate in candidates if candidate.rejected_reason]
    log(f"brain: only {len(fresh)} fresh of {len(candidates)}; asking again")
    retry_prompt = candidates_prompt(context, n=n, rejects=rejects, reach=reach, lean=lean, misses=misses)
    more = ask_brain(backend, retry_prompt, timeout)
    mark_rejects(more, store, seen_keys)
    return candidates + more
