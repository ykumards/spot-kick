"""Context, not memory. The prompt is built from a fixed set of capped store queries, so it is ~20 lines whether
the history holds 10 plays or 10,000. The Brain sees names and words; never vectors, never the whole log.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REACH_TEXT = {
    "near": "a small step: same mood, one notch more adventurous",
    "adjacent": "a real kick: leave the current pocket for an adjacent one, keep one thread of continuity",
    "far": (
        "a boot: a distant but musically defensible destination, the kind of thing this listener would never be "
        "recommended"
    ),
}

DIG = {
    0: "",
    1: (
        "Skip the obvious: no song that would sit on a 'best of the genre' list, no artist's single most famous track. "
        "Prefer album cuts and artists of moderate fame."
    ),
    2: (
        "Go deep: nothing you'd expect on a mainstream playlist. Prefer lesser-known artists, deep cuts, regional "
        "scenes, reissues — a song a serious fan of the direction would name, not a tourist."
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
SOURCE_TAG = {"spotify": "", "kick": " [kick]", "minime": " [mini-me]", "user": ""}
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
    taste: list[str] = field(default_factory=list)  # taste modes in words, when the Mind has them

    @classmethod
    def from_store(cls, store, *, taste: list[str] | None = None) -> Context:
        return cls(
            recent=store.recent(RECENT_PLAYS),
            top_recent=store.top_artists(days=TOP_RECENT_DAYS, n=TOP_ARTISTS),
            top_all=store.top_artists(n=TOP_ARTISTS),
            loved=store.loved(LOVED_TRACKS),
            rejected=store.rejected(days=REJECTED_DAYS, n=REJECTED_TRACKS),
            directions=store.directions(KICKED_DIRECTIONS),
            kicked_artists=store.kicked_artists(KICKED_ARTISTS),
            taste=list(taste or []),
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
        if self.taste:
            lines.append("Their taste, in modes: " + "; ".join(self.taste))
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
    """The listener's lean bounds every pick; the reach still varies inside it (a 'far' pick is far *within* it)."""
    cleaned = " ".join(lean.split())
    instruction = f'The listener asked for this lean, and every pick must stay inside it: "{cleaned}".'
    return instruction + " Vary the reach inside it."


def candidates_prompt(
    context: Context,
    *,
    n: int = 6,
    dig: int = 1,
    direction_hint: str | None = None,
    rejects: list[str] | None = None,
    reach: str | None = None,
    lean: str | None = None,
) -> str:
    """One call, several graded candidates in distinct directions. A separate step measures them and picks.

    With `reach`, every candidate is asked for at that one reach: the top-up for a band the pool has run out of.
    With `lean`, the listener's own words (a mood, a language, an era) bound every pick."""
    per_reach = max(1, n // REACH_GROUPS)
    head = (
        "You are proposing songs to kick a listener's Spotify recommendations somewhere new. A separate step measures "
        "each candidate's audio distance from their current listening and plays the one that lands where the listener "
        "aimed; you only name real, existing songs."
    )
    if direction_hint:
        ask = (
            f'Propose {n} real songs that continue in ONE direction: "{direction_hint}". Order them so each is a '
            "plausible next step from the previous; the first should be the closest to that direction's starting song. "
            "Label each 'reach' as adjacent."
        )
    elif reach:
        ask = (
            f"Propose {n} real songs, each in a DIFFERENT direction, all labelled '{reach}' ({REACH_TEXT[reach]}). "
            "Earlier picks at this reach measured closer to the listener than intended, so lean further out."
        )
    else:
        near_ask = f"{per_reach} labelled 'near' ({REACH_TEXT['near']})"
        adjacent_ask = f"{per_reach} 'adjacent' ({REACH_TEXT['adjacent']})"
        far_ask = f"{per_reach} 'far' ({REACH_TEXT['far']})"
        total = REACH_GROUPS * per_reach
        ask = f"Propose {total} real songs, each in a DIFFERENT direction: {near_ask}, {adjacent_ask}, {far_ask}."
    rules = (
        "Spotify's autoplay will continue FROM the chosen song, so each needs a strong, coherent identity a "
        "recommender can build a queue around — not a novelty, not a one-off. Never repeat anything listed above. "
        + DIG[dig]
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
