"""Resolution of (artist, title) to a Spotify track URI.

The first search hit whose artist and title match the request is used. A hit for a different song would be
measured as the wrong audio, so no match means no track.
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
    """Normalise a name to lower-case ASCII words with accents removed."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    return normalize_name(ascii_only)


def names_match(want: str, got: str) -> bool:
    """Return True when either name contains the other, or the first words of ``want`` appear in ``got``."""
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
    """Return the first Spotify hit that matches the song, or None.

    An unreachable Spotify is logged and treated as no match.
    """
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
