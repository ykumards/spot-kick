"""30-second preview lookup through the iTunes Search API, which needs no authentication."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

from ..names import normalize_name

SEARCH = "https://itunes.apple.com/search"
SEARCH_LIMIT = 8
SEARCH_TIMEOUT_S = 15


class HttpResponse(Protocol):
    """The part of ``requests.Response`` the lookup reads."""

    @property
    def status_code(self) -> int: ...

    def json(self) -> dict: ...

    def raise_for_status(self) -> None: ...


class HttpClient(Protocol):
    """Anything with a ``requests``-style ``get``: the ``requests`` module, a ``Session``, or a test fake."""

    def get(
        self, url: str, *, params: dict | None = ..., headers: dict | None = ..., timeout: float | None = ...
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class Preview:
    artist: str
    title: str
    album: str | None
    preview_url: str | None
    duration_s: float | None


def match_strength(wanted: str, found: str) -> int:
    """Return 2 for an exact match, 1 when one name contains the other, 0 otherwise. Inputs must be normalised."""
    if wanted == found:
        return 2
    if wanted in found or found in wanted:
        return 1
    return 0


def rank_key(result: dict, wanted_artist: str, wanted_title: str) -> tuple[int, int, int]:
    """Return a sort key: artist match, then title match, then shortest title (album versions over remixes)."""
    found_artist = normalize_name(result.get("artistName", ""))
    found_title = normalize_name(result.get("trackName", ""))
    artist_strength = match_strength(wanted_artist, found_artist)
    title_strength = match_strength(wanted_title, found_title)
    return (artist_strength, title_strength, -len(found_title))


def to_preview(result: dict) -> Preview:
    duration_s = (result.get("trackTimeMillis") or 0) / 1000.0 or None
    return Preview(
        artist=result["artistName"],
        title=result["trackName"],
        album=result.get("collectionName"),
        preview_url=result.get("previewUrl"),
        duration_s=duration_s,
    )


def lookup(artist: str, title: str, *, country: str = "us", session: HttpClient | None = None) -> Preview | None:
    """Return the best-matching result with a preview, or None when neither artist nor title matches."""
    http = session or requests
    params = {"term": f"{artist} {title}", "entity": "song", "limit": SEARCH_LIMIT, "country": country}
    response = http.get(SEARCH, params=params, timeout=SEARCH_TIMEOUT_S)
    response.raise_for_status()
    results = [result for result in response.json().get("results", []) if result.get("previewUrl")]
    if not results:
        return None
    wanted_artist = normalize_name(artist)
    wanted_title = normalize_name(title)
    best = max(results, key=lambda result: rank_key(result, wanted_artist, wanted_title))
    artist_strength, title_strength, _ = rank_key(best, wanted_artist, wanted_title)
    if artist_strength == 0 or title_strength == 0:
        return None  # a preview for another song would corrupt the ruler
    return to_preview(best)
