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

OSASCRIPT_TIMEOUT_S = 15
CONFIRM_POLL_S = 1.0
DEFAULT_CONFIRM_TIMEOUT_S = 8.0
DEFAULT_UNMUTE_VOLUME = 60
MIN_VOLUME = 0
MAX_VOLUME = 100

# Positions of the tab-separated fields NOW_PLAYING_SCRIPT returns.
FIELD_NAME = 0
FIELD_ARTIST = 1
FIELD_ALBUM = 2
FIELD_DURATION_MS = 3
FIELD_POSITION_S = 4
FIELD_URI = 5
FIELD_POPULARITY = 6
FIELD_ARTWORK_URL = 7
REQUIRED_FIELDS = 6

NOT_RUNNING_MARKER = "__NOT_RUNNING__"
IS_RUNNING_SCRIPT = 'application "Spotify" is running'
STATE_SCRIPT = 'tell application "Spotify" to get player state as string'
NOW_PLAYING_SCRIPT = (
    'tell application "Spotify" to return (name of current track) & tab & (artist of current track) & tab & '
    '(album of current track) & tab & (duration of current track) & tab & (player position) & tab & '
    '(id of current track) & tab & (popularity of current track) & tab & (artwork url of current track)'
)
NEXT_TRACK_SCRIPT = 'tell application "Spotify" to next track'
PLAYPAUSE_SCRIPT = 'tell application "Spotify" to playpause'
GET_VOLUME_SCRIPT = 'tell application "Spotify" to get sound volume'


def play_script(uri: str) -> str:
    return f'tell application "Spotify" to play track "{uri}"'


def set_volume_script(level: int) -> str:
    return f'tell application "Spotify" to set sound volume to {level}'


NOT_RUNNING_MESSAGE = "Spotify isn't running"


class PlayerError(RuntimeError):
    pass


def guarded(script: str) -> str:
    """`tell application "Spotify"` launches Spotify when it is not running — and a Spotify launched that way in the
    background sits stopped and ignores play requests, while the observer would relaunch it seconds after the listener
    quit it. `application "Spotify" is running` neither launches nor needs a permission, and checking it in the same
    script as the command leaves no window for Spotify to quit in between."""
    return (
        f"if {IS_RUNNING_SCRIPT} then\n"
        f"    {script}\n"
        "else\n"
        f'    return "{NOT_RUNNING_MARKER}"\n'
        "end if"
    )


def run_applescript(script: str) -> str:
    """Run one AppleScript line through osascript and return its trimmed stdout."""
    completed = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=OSASCRIPT_TIMEOUT_S, check=False
    )
    if completed.returncode != 0:
        raise PlayerError(completed.stderr.strip())
    return completed.stdout.strip()


@dataclass(frozen=True)
class Track:
    name: str
    artist: str
    album: str
    duration_s: float
    position_s: float
    uri: str = ""
    popularity: int | None = None
    artwork_url: str | None = None

    @property
    def label(self) -> str:
        return f"{self.artist} — {self.name}"


def to_uri(uri: str) -> str | None:
    """Canonical `spotify:track:<id>`, or None for anything that is not a track URI."""
    match = URI_RE.search(uri or "")
    if match is None:
        return None
    return f"spotify:track:{match.group(1)}"


def tell_spotify(script: str) -> str:
    """Every script addressed to Spotify goes through here, wrapped in the running guard (see `guarded`)."""
    result = run_applescript(guarded(script))
    if result == NOT_RUNNING_MARKER:
        raise PlayerError(NOT_RUNNING_MESSAGE)
    return result


def state() -> str:
    """playing | paused | stopped"""
    return tell_spotify(STATE_SCRIPT)


def parse_number(text: str) -> float:
    """AppleScript numbers as floats; anything unparsable counts as zero."""
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_now_playing(raw: str) -> Track | None:
    """Turn NOW_PLAYING_SCRIPT's tab-separated line into a Track. None if the line is too short to be one."""
    fields = raw.split("\t")
    if len(fields) < REQUIRED_FIELDS:
        return None
    popularity = None
    if len(fields) > FIELD_POPULARITY and fields[FIELD_POPULARITY] != "":
        popularity = int(parse_number(fields[FIELD_POPULARITY]))
    artwork_url = None
    if len(fields) > FIELD_ARTWORK_URL and fields[FIELD_ARTWORK_URL].startswith("http"):
        artwork_url = fields[FIELD_ARTWORK_URL]
    return Track(
        name=fields[FIELD_NAME],
        artist=fields[FIELD_ARTIST],
        album=fields[FIELD_ALBUM],
        duration_s=parse_number(fields[FIELD_DURATION_MS]) / 1000.0,
        position_s=parse_number(fields[FIELD_POSITION_S]),
        uri=fields[FIELD_URI],
        popularity=popularity,
        artwork_url=artwork_url,
    )


def now_playing() -> Track | None:
    if state() == "stopped":
        return None
    return parse_now_playing(tell_spotify(NOW_PLAYING_SCRIPT))


def play(uri: str) -> None:
    canonical = to_uri(uri)
    if not canonical:
        raise ValueError(f"not a Spotify track: {uri}")
    tell_spotify(play_script(canonical))


def play_and_confirm(uri: str, *, timeout_s: float = DEFAULT_CONFIRM_TIMEOUT_S) -> Track:
    """Issue the play, then read back what is actually playing. Raises if it isn't the requested URI."""
    play(uri)
    wanted = to_uri(uri)
    deadline = time.time() + timeout_s
    current = None
    while time.time() < deadline:
        time.sleep(CONFIRM_POLL_S)
        try:
            current = now_playing()
        except PlayerError:
            current = None
        if current is not None and current.uri == wanted and state() == "playing":
            return current
    if current is None:
        has = "nothing"
    else:
        has = current.uri
    raise PlayerError(f"asked for {wanted}, player has {has}")


def next_track() -> None:
    tell_spotify(NEXT_TRACK_SCRIPT)


def playpause() -> None:
    tell_spotify(PLAYPAUSE_SCRIPT)


def volume() -> int:
    return int(float(tell_spotify(GET_VOLUME_SCRIPT) or 0))


def set_volume(level: int) -> None:
    clamped = max(MIN_VOLUME, min(MAX_VOLUME, int(level)))
    tell_spotify(set_volume_script(clamped))


_last_volume = DEFAULT_UNMUTE_VOLUME


def toggle_mute() -> bool:
    """Returns True if now muted."""
    global _last_volume
    current = volume()
    if current > 0:
        _last_volume = current
        set_volume(0)
        return True
    set_volume(_last_volume or DEFAULT_UNMUTE_VOLUME)
    return False
