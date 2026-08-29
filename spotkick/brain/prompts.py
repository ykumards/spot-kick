"""Context, not memory. The prompt is built from a fixed set of capped store queries, so it is ~20 lines whether
the history holds 10 plays or 10,000. The Brain sees names and words; never vectors, never the whole log.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REACH_TEXT = {
    "near": "a small step: same mood, one notch more adventurous",
    "adjacent": "a real kick: leave the current pocket for an adjacent one, keep one thread of continuity",
    "far": "a boot: a distant but musically defensible destination, the kind of thing this listener would never be recommended",
}

DIG = {
    0: "",
    1: "Skip the obvious: no song that would sit on a 'best of the genre' list, no artist's single most famous track. "
       "Prefer album cuts and artists of moderate fame.",
    2: "Go deep: nothing you'd expect on a mainstream playlist. Prefer lesser-known artists, deep cuts, regional scenes, "
       "reissues — a song a serious fan of the direction would name, not a tourist.",
}

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
        "reach": {"type": "string", "enum": ["near", "adjacent", "far"]},
        "direction": {"type": "string"}, "artist": {"type": "string"}, "title": {"type": "string"},
        "why": {"type": "string"}, "spotify_uri": {"type": "string"}},
        "required": ["reach", "direction", "artist", "title", "why", "spotify_uri"], "additionalProperties": False}}},
    "required": ["candidates"], "additionalProperties": False,
}


@dataclass
class Context:
    recent: list[dict] = field(default_factory=list)        # {artist, title, source, kind}
    top_recent: list[tuple[str, int]] = field(default_factory=list)
    top_all: list[tuple[str, int]] = field(default_factory=list)
    loved: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    kicked_artists: list[str] = field(default_factory=list)
    taste: list[str] = field(default_factory=list)           # taste modes in words, when the Mind has them

    @classmethod
    def from_store(cls, store, *, taste: list[str] | None = None) -> Context:
        return cls(recent=store.recent(12), top_recent=store.top_artists(days=30, n=10), top_all=store.top_artists(n=10),
                   loved=store.loved(8), rejected=store.rejected(days=14, n=8), directions=store.directions(10),
                   kicked_artists=store.kicked_artists(25), taste=list(taste or []))

    def lines(self) -> list[str]:
        out = []
        if self.recent:
            src = {"spotify": "", "kick": " [kick]", "minime": " [mini-me]", "user": ""}
            out.append("Last plays, most recent first:")
            for r in self.recent:
                tag = " (skipped)" if r["kind"] == "skip" else " (partial)" if r["kind"] == "partial" else ""
                out.append(f"- {r['artist']} — {r['title']}{src.get(r['source'], '')}{tag}")
        if self.top_recent:
            out.append("Most played artists, last 30 days: " + ", ".join(f"{a} ({n})" for a, n in self.top_recent))
        if self.top_all and [a for a, _ in self.top_all] != [a for a, _ in self.top_recent]:
            out.append("Most played artists, all time: " + ", ".join(f"{a} ({n})" for a, n in self.top_all))
        if self.taste:
            out.append("Their taste, in modes: " + "; ".join(self.taste))
        if self.loved:
            out.append("Loved: " + "; ".join(self.loved))
        if self.rejected:
            out.append("Skipped or disliked recently — not this vein: " + "; ".join(self.rejected))
        if self.directions:
            out.append("Directions already kicked toward (choose different ones): " + "; ".join(self.directions))
        if self.kicked_artists:
            out.append("Artists already kicked to, avoid: " + ", ".join(self.kicked_artists))
        return out


def candidates_prompt(ctx: Context, *, n: int = 6, dig: int = 1, direction_hint: str | None = None,
                      rejects: list[str] | None = None) -> str:
    """One call, several graded candidates in distinct directions. A separate step measures them and picks."""
    per = max(1, n // 3)
    head = ("You are proposing songs to kick a listener's Spotify recommendations somewhere new. A separate step measures "
            "each candidate's audio distance from their current listening and plays the one that lands where the listener "
            "aimed; you only name real, existing songs.")
    if direction_hint:
        ask = (f"Propose {n} real songs that continue in ONE direction: \"{direction_hint}\". Order them so each is a plausible "
               "next step from the previous; the first should be the closest to that direction's starting song. Label each "
               "'reach' as adjacent.")
    else:
        ask = (f"Propose {3 * per} real songs, each in a DIFFERENT direction: {per} labelled 'near' ({REACH_TEXT['near']}), "
               f"{per} 'adjacent' ({REACH_TEXT['adjacent']}), {per} 'far' ({REACH_TEXT['far']}).")
    rules = ("Spotify's autoplay will continue FROM the chosen song, so each needs a strong, coherent identity a recommender "
             "can build a queue around — not a novelty, not a one-off. Never repeat anything listed above. " + DIG[dig])
    tail = ("For each: a 3-6 word direction, the real artist and title, a one-line why, and the Spotify track URI "
            "(spotify:track:<22 chars>) of the canonical studio recording if you are confident of it, else an empty string. "
            "No links, no markdown.")
    parts = [head, "", *ctx.lines(), ""]
    if rejects:
        parts += ["These were proposed before and rejected (already known to the listener) — do not propose them again: "
                  + "; ".join(rejects), ""]
    parts += [ask, rules, tail]
    return "\n".join(parts)
