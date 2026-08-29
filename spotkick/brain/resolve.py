"""(artist, title) → a Spotify track URI we can trust, without Spotify credentials.

1. The Brain names a URI from memory (fast, sometimes hallucinated).
2. Validate with Spotify's public oEmbed endpoint: a real id returns the track's title, a fake one 404s.
   The title must also resemble what we asked for.
3. If that fails, a `searcher(artist, title) -> uri | None` (an LLM with web search, or the Spotify Web API)
   gets one try, validated the same way.
4. The player reads back the current track after the play; that check lives in `player.spotify.play_and_confirm`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import requests

OEMBED = "https://open.spotify.com/oembed"
URI_RE = re.compile(r"spotify:track:([A-Za-z0-9]{22})")

Searcher = Callable[[str, str], str | None]


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


def titles_match(want: str, got: str) -> bool:
    want, got = _norm(want), _norm(got)
    if not want or not got:
        return False
    head = " ".join(want.split(" ")[:3])
    return want in got or got in want or head in got


def oembed_title(uri: str, *, session=None) -> str | None:
    """Title Spotify reports for this id, or None if the id doesn't exist."""
    m = URI_RE.search(uri or "")
    if not m:
        return None
    http = session or requests
    r = http.get(OEMBED, params={"url": f"https://open.spotify.com/track/{m.group(1)}"}, timeout=10)
    if r.status_code != 200:
        return None
    return r.json().get("title", "") or None


def validate(uri: str | None, title: str, *, session=None) -> bool:
    got = oembed_title(uri or "", session=session)
    return got is not None and titles_match(title, got)


@dataclass(frozen=True)
class Resolved:
    uri: str
    how: str          # memory | search
    oembed_title: str


def resolve(artist: str, title: str, candidate: str | None = None, *, searcher: Searcher | None = None,
            session=None, log=lambda m: None) -> Resolved | None:
    if candidate:
        got = oembed_title(candidate, session=session)
        if got and titles_match(title, got):
            return Resolved(URI_RE.search(candidate).group(0), "memory", got)
        log(f"resolve: '{candidate}' is not {artist} — {title} (oEmbed: {got!r}); searching")
    if searcher is None:
        return None
    uri = searcher(artist, title)
    if uri:
        got = oembed_title(uri, session=session)
        if got and titles_match(title, got):
            return Resolved(URI_RE.search(uri).group(0), "search", got)
        log(f"resolve: search gave '{uri}' (oEmbed: {got!r}), not validated")
    return None
