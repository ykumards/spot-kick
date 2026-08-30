"""Context, not memory. The prompt is built from a fixed set of capped store queries, so it is ~20 lines whether
the history holds 10 plays or 10,000. The Brain sees names and words; never vectors, never the whole log.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# One knob: how hard the kick is. A harder kick is farther in sound *and* deeper below the surface, because that is
# what "farther" means to a listener: a small step may be a song they half know; a boot is one they would never meet.
REACH_TEXT = {
    "near": "a small step: same mood, one notch more adventurous; a known song is fine if it fits",
    "adjacent": (
        "a real kick: leave the current pocket for an adjacent one, keep one thread of continuity; skip the obvious — "
        "no 'best of the genre' pick, no artist's single most famous track, prefer album cuts and moderate fame"
    ),
    "far": (
        "a boot: a distant but musically defensible destination, the kind of thing this listener would never be "
        "recommended; go deep — lesser-known artists, deep cuts, regional scenes, reissues, a song a serious fan of "
        "the direction would name, not a tourist"
    ),
}

CANDIDATE_FIELDS = ["reach", "direction", "artist", "title", "why"]

CANDIDATE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "reach": {"type": "string", "enum": ["near", "adjacent", "far"]},
        "direction": {"type": "string"},
        "artist": {"type": "string"},
        "title": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": CANDIDATE_FIELDS,
    "additionalProperties": False,
}

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": CANDIDATE_ITEM_SCHEMA}},
    "required": ["candidates"],
    "additionalProperties": False,
}

# How many of each list the prompt carries, so its size is bounded regardless of history length.
RECENT_PLAYS = 12
TOP_ARTISTS = 10
TOP_RECENT_DAYS = 30
LOVED_TRACKS = 8
REJECTED_TRACKS = 8
REJECTED_DAYS = 14
KICKED_DIRECTIONS = 10
KICKED_ARTISTS = 25

# Annotations on each recent play, keyed by the event's source and kind.
SOURCE_TAG = {"spotify": "", "kick": " [kick]"}
KIND_TAG = {"skip": " (skipped)", "partial": " (partial)"}

REACH_GROUPS = 3


@dataclass
class Context:
    recent: list[dict] = field(default_factory=list)  # {artist, title, source, kind}
    top_recent: list[tuple[str, int]] = field(default_factory=list)
    top_all: list[tuple[str, int]] = field(default_factory=list)
    loved: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    kicked_artists: list[str] = field(default_factory=list)

    @classmethod
    def from_store(cls, store) -> Context:
        return cls(
            recent=store.recent(RECENT_PLAYS),
            top_recent=store.top_artists(days=TOP_RECENT_DAYS, n=TOP_ARTISTS),
            top_all=store.top_artists(n=TOP_ARTISTS),
            loved=store.loved(LOVED_TRACKS),
            rejected=store.rejected(days=REJECTED_DAYS, n=REJECTED_TRACKS),
            directions=store.directions(KICKED_DIRECTIONS),
            kicked_artists=store.kicked_artists(KICKED_ARTISTS),
        )

    def recent_play_lines(self) -> list[str]:
        """One bullet per recent play, annotated with where it came from and whether it was skipped."""
        lines = ["Last plays, most recent first:"]
        for play in self.recent:
            source_tag = SOURCE_TAG.get(play["source"], "")
            kind_tag = KIND_TAG.get(play["kind"], "")
            lines.append(f"- {play['artist']} — {play['title']}{source_tag}{kind_tag}")
        return lines

    def lines(self) -> list[str]:
        lines: list[str] = []
        if self.recent:
            lines.extend(self.recent_play_lines())
        if self.top_recent:
            lines.append("Most played artists, last 30 days: " + artist_counts(self.top_recent))
        if self.top_all and artist_names(self.top_all) != artist_names(self.top_recent):
            lines.append("Most played artists, all time: " + artist_counts(self.top_all))
        if self.loved:
            lines.append("Loved: " + "; ".join(self.loved))
        if self.rejected:
            lines.append("Skipped or disliked recently — not this vein: " + "; ".join(self.rejected))
        if self.directions:
            lines.append("Directions already kicked toward (choose different ones): " + "; ".join(self.directions))
        if self.kicked_artists:
            lines.append("Artists already kicked to, avoid: " + ", ".join(self.kicked_artists))
        return lines


def artist_counts(ranked: list[tuple[str, int]]) -> str:
    return ", ".join(f"{artist} ({count})" for artist, count in ranked)


def artist_names(ranked: list[tuple[str, int]]) -> list[str]:
    return [artist for artist, _count in ranked]


def lean_line(lean: str) -> str:
    """The listener's lean bounds every pick and outranks the rest of the prompt: 'top 50' must not lose to a reach's
    'nothing famous', and a 'far' pick is far *within* the lean, not outside it."""
    cleaned = " ".join(lean.split())
    instruction = f'The listener asked for this lean, and every pick must stay inside it: "{cleaned}".'
    priority = (
        " Take it literally. Where it conflicts with anything else here — how famous or obscure a pick should be, "
        "how far a reach goes — the lean wins. Vary the reach inside it."
    )
    return instruction + priority


MISSES_SHOWN = 6
BAND_WORD = {"tap": "a small step", "kick": "a kick", "boot": "a boot"}


def misses_line(misses: list[dict], reach: str) -> str:
    """What the audio ruler made of the last picks at this reach: the brain named them, the measurement disagreed.
    Telling it exactly which songs landed where is what lets it correct instead of repeating itself."""
    landed = "; ".join(
        f"{miss['artist']} — {miss['title']} measured as {BAND_WORD.get(miss['band'], miss['band'])}"
        for miss in misses[-MISSES_SHOWN:]
    )
    return (
        f"Your earlier picks for '{reach}' did not land: {landed}. Those are too close in sound to what the listener "
        "plays. Go decisively further from all of them — another tradition, era, instrumentation, tempo or language — "
        "and do not propose them again."
    )


def candidates_prompt(
    context: Context,
    *,
    n: int = 6,
    rejects: list[str] | None = None,
    reach: str | None = None,
    lean: str | None = None,
    misses: list[dict] | None = None,
) -> str:
    """One call, several graded candidates in distinct directions. A separate step measures them and picks.

    With `reach`, every candidate is asked for at that one reach: the top-up for a band the pool has run out of.
    With `misses`, the earlier picks at that reach and where each actually measured, so the brain corrects.
    With `lean`, the listener's own words (a mood, a language, an era) bound every pick."""
    per_reach = max(1, n // REACH_GROUPS)
    head = (
        "You are proposing songs to kick a listener's Spotify recommendations somewhere new. A separate step measures "
        "each candidate's audio distance from their current listening and plays the one that lands where the listener "
        "aimed; you only name real, existing songs."
    )
    if reach:
        ask = f"Propose {n} real songs, each in a DIFFERENT direction, all labelled '{reach}' ({REACH_TEXT[reach]})."
        if misses:
            ask += " " + misses_line(misses, reach)
    else:
        near_ask = f"{per_reach} labelled 'near' ({REACH_TEXT['near']})"
        adjacent_ask = f"{per_reach} 'adjacent' ({REACH_TEXT['adjacent']})"
        far_ask = f"{per_reach} 'far' ({REACH_TEXT['far']})"
        total = REACH_GROUPS * per_reach
        ask = f"Propose {total} real songs, each in a DIFFERENT direction: {near_ask}, {adjacent_ask}, {far_ask}."
    rules = (
        "Spotify's autoplay will continue FROM the chosen song, so each needs a strong, coherent identity a "
        "recommender can build a queue around — not a novelty, not a one-off. Never repeat anything listed above."
    )
    tail = "For each: a 3-6 word direction, the real artist and title, and a one-line why. No links, no markdown."
    parts = [head, "", *context.lines(), ""]
    if lean:
        parts.extend([lean_line(lean), ""])
    if rejects:
        rejects_line = (
            "These were proposed before and rejected (already known to the listener) — do not propose them again: "
            + "; ".join(rejects)
        )
        parts.extend([rejects_line, ""])
    parts.extend([ask, rules, tail])
    return "\n".join(parts)
