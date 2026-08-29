"""The hands: the Spotify desktop app over AppleScript. The only module that runs osascript.

Spotify plays any catalog track by URI and reports name / artist / album / duration / position / uri /
popularity. We do not touch its window; if it pops forward on a play, so be it.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass

BUNDLE = "com.spotify.client"
URI_RE = re.compile(r"spotify:track:([A-Za-z0-9]{22})")
URL_RE = re.compile(r"open\.spotify\.com/(?:intl-[a-z]+/)?track/([A-Za-z0-9]{22})")


class PlayerError(RuntimeError):
    pass


def _osa(script: str) -> str:
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15, check=False)
    if out.returncode != 0:
        raise PlayerError(out.stderr.strip())
    return out.stdout.strip()


@dataclass(frozen=True)
class Track:
    name: str
    artist: str
    album: str
    duration_s: float
    position_s: float
    uri: str = ""
    popularity: int | None = None

    @property
    def label(self) -> str:
        return f"{self.artist} — {self.name}"

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.duration_s - self.position_s)


def to_uri(link_or_uri: str) -> str | None:
    m = URI_RE.search(link_or_uri or "") or URL_RE.search(link_or_uri or "")
    return f"spotify:track:{m.group(1)}" if m else None


def is_running() -> bool:
    return _osa('tell application "System Events" to (name of processes) contains "Spotify"') == "true"


def state() -> str:
    """playing | paused | stopped"""
    return _osa('tell application "Spotify" to get player state as string')


NOW_PLAYING_SCRIPT = ('tell application "Spotify" to return (name of current track) & tab & (artist of current track) & tab & '
                      '(album of current track) & tab & (duration of current track) & tab & (player position) & tab & '
                      '(id of current track) & tab & (popularity of current track)')


def parse_now_playing(raw: str) -> Track | None:
    parts = raw.split("\t")
    if len(parts) < 6:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return 0.0

    pop = int(num(parts[6])) if len(parts) > 6 and parts[6] != "" else None
    return Track(name=parts[0], artist=parts[1], album=parts[2], duration_s=num(parts[3]) / 1000.0, position_s=num(parts[4]),
                 uri=parts[5], popularity=pop)


def now_playing() -> Track | None:
    if state() == "stopped":
        return None
    return parse_now_playing(_osa(NOW_PLAYING_SCRIPT))


def play(uri: str) -> None:
    u = to_uri(uri)
    if not u:
        raise ValueError(f"not a Spotify track: {uri}")
    _osa(f'tell application "Spotify" to play track "{u}"')


def play_and_confirm(uri: str, *, timeout_s: float = 8.0) -> Track:
    """Issue the play, then read back what is actually playing. Raises if it isn't the requested URI."""
    play(uri)
    want = to_uri(uri)
    t_end, t = time.time() + timeout_s, None
    while time.time() < t_end:
        time.sleep(1.0)
        try:
            t = now_playing()
        except PlayerError:
            t = None
        if t is not None and t.uri == want and state() == "playing":
            return t
    raise PlayerError(f"asked for {want}, player has {t.uri if t else 'nothing'}")


def next_track() -> None:
    _osa('tell application "Spotify" to next track')


def playpause() -> None:
    _osa('tell application "Spotify" to playpause')


def volume() -> int:
    return int(float(_osa('tell application "Spotify" to get sound volume') or 0))


def set_volume(v: int) -> None:
    _osa(f'tell application "Spotify" to set sound volume to {max(0, min(100, int(v)))}')


_last_volume = 60


def toggle_mute() -> bool:
    """Returns True if now muted."""
    global _last_volume
    v = volume()
    if v > 0:
        _last_volume = v
        set_volume(0)
        return True
    set_volume(_last_volume or 60)
    return False
