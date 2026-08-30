"""(artist, title) → a Spotify track URI we can trust.

The brain names songs; Spotify's search names ids. The first hit whose artist and title resemble what was asked
for wins; a hit for another song would put the wrong audio under the ruler, so no match means no track.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from ..names import normalize_name
from ..player.spotify_api import SpotifyAPI, SpotifyAPIError

HEAD_WORDS = 3

Logger = Callable[[str], None]


def ignore_log(message: str) -> None:
    return None


def normalize(name: str) -> str:
    """Lower-case ASCII words: accents dropped, so Spotify's 'Nètsanèt' matches the brain's 'Netsanet'."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    return normalize_name(ascii_only)


def names_match(want: str, got: str) -> bool:
    """Loose containment either way, or the first few words of what we wanted appear in what Spotify reports."""
    wanted = normalize(want)
    reported = normalize(got)
    if not wanted or not reported:
        return False
    wanted_head = " ".join(wanted.split(" ")[:HEAD_WORDS])
    return wanted in reported or reported in wanted or wanted_head in reported


@dataclass(frozen=True)
class Resolved:
    uri: str
    artist: str   # as Spotify names them
    title: str


def resolve(artist: str, title: str, api: SpotifyAPI, *, log: Logger = ignore_log) -> Resolved | None:
    """Spotify's first hit that is actually this song, or None. Spotify being unreachable is logged and counts as
    no track — the candidate is skipped, the kick goes on."""
    try:
        hits = api.search_tracks(artist, title)
    except SpotifyAPIError as error:
        log(f"resolve: {error}")
        return None
    for hit in hits:
        if names_match(artist, hit.artist) and names_match(title, hit.title):
            return Resolved(hit.uri, hit.artist, hit.title)
    if hits:
        log(f"resolve: no hit for {artist} — {title} is that song (first: {hits[0].artist} — {hits[0].title})")
    return None
