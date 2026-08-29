"""Ask the Brain for candidates; reject the ones the store already knows; ask once more if the set came back thin.
The output is names only — resolving, embedding, and choosing happen elsewhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import CANDIDATES_SCHEMA, Context, candidates_prompt

URI_RE = re.compile(r"spotify:track:[A-Za-z0-9]{22}")


@dataclass
class Candidate:
    reach: str
    direction: str
    artist: str
    title: str
    why: str
    spotify_uri: str | None
    rejected_reason: str | None = None

    def as_row(self) -> dict:
        return {"reach": self.reach, "direction": self.direction, "artist": self.artist, "title": self.title, "why": self.why,
                "spotify_uri": self.spotify_uri, "rejected_reason": self.rejected_reason}


def _strip(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text or "")
    return re.sub(r"https?://\S+", "", text).strip()


def _parse(raw: list[dict]) -> list[Candidate]:
    out = []
    for c in raw:
        uri = URI_RE.search(c.get("spotify_uri") or "")
        artist, title = (c.get("artist") or "").strip(), (c.get("title") or "").strip()
        if not artist or not title:
            continue
        out.append(Candidate(reach=c.get("reach") or "adjacent", direction=_strip(c.get("direction", "")), artist=artist, title=title,
                             why=_strip(c.get("why", "")), spotify_uri=uri.group(0) if uri else None))
    return out


def propose(backend, store, *, n: int = 6, dig: int = 1, direction_hint: str | None = None, taste: list[str] | None = None,
            min_fresh: int = 3, timeout: int = 240, log=lambda m: None) -> list[Candidate]:
    """Candidates the listener hasn't already played/kicked/picked, with the rejects kept (marked) for the log."""
    ctx = Context.from_store(store, taste=taste)
    prompt = candidates_prompt(ctx, n=n, dig=dig, direction_hint=direction_hint)
    cands = _parse(backend.complete_json(prompt, CANDIDATES_SCHEMA, timeout=timeout).get("candidates", []))
    seen_here: set[str] = set()
    for c in cands:
        key = f"{c.artist.lower()}|{c.title.lower()}"
        if key in seen_here:
            c.rejected_reason = "duplicate in set"
        elif store.seen(c.artist, c.title):
            c.rejected_reason = "already known"
        seen_here.add(key)
    fresh = [c for c in cands if not c.rejected_reason]
    if len(fresh) < min_fresh:
        rejects = [f"{c.artist} — {c.title}" for c in cands if c.rejected_reason]
        log(f"brain: only {len(fresh)} fresh of {len(cands)}; asking again")
        more = _parse(backend.complete_json(candidates_prompt(ctx, n=n, dig=dig, direction_hint=direction_hint, rejects=rejects),
                                            CANDIDATES_SCHEMA, timeout=timeout).get("candidates", []))
        for c in more:
            key = f"{c.artist.lower()}|{c.title.lower()}"
            if key in seen_here:
                c.rejected_reason = "duplicate in set"
            elif store.seen(c.artist, c.title):
                c.rejected_reason = "already known"
            seen_here.add(key)
        cands += more
    return cands
