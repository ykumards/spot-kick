"""30-second previews from the iTunes Search API. No auth; this is where the audio for the ruler comes from."""
from __future__ import annotations

import re
from dataclasses import dataclass

import requests

SEARCH = "https://itunes.apple.com/search"


@dataclass(frozen=True)
class Preview:
    artist: str
    title: str
    album: str | None
    itunes_id: int
    preview_url: str | None
    duration_s: float | None
    genre: str | None


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


def lookup(artist: str, title: str, *, country: str = "us", session: requests.Session | None = None) -> Preview | None:
    """Best match for (artist, title) with a preview. Prefers exact-ish artist and title matches, then the shortest title
    (album versions over remixes/live cuts)."""
    http = session or requests
    r = http.get(SEARCH, params={"term": f"{artist} {title}", "entity": "song", "limit": 8, "country": country}, timeout=15)
    r.raise_for_status()
    results = [x for x in r.json().get("results", []) if x.get("previewUrl")]
    if not results:
        return None
    a, t = _norm(artist), _norm(title)

    def score(x):
        xa, xt = _norm(x.get("artistName", "")), _norm(x.get("trackName", ""))
        return (2 * (a == xa) + (a in xa or xa in a), 2 * (t == xt) + (t in xt or xt in t), -len(xt))

    best = max(results, key=score)
    if score(best)[0] == 0 and score(best)[1] == 0:
        return None  # nothing resembling the request
    return Preview(artist=best["artistName"], title=best["trackName"], album=best.get("collectionName"), itunes_id=best["trackId"],
                   preview_url=best.get("previewUrl"), duration_s=(best.get("trackTimeMillis") or 0) / 1000.0 or None,
                   genre=best.get("primaryGenreName"))
