"""The no-kick control: the only way to separate "Spotify built a new queue from the kick" from "Spotify resumed
a queue it already had".

Two arms, same seed, run back to back with Spotify muted:
    control:  play the seed, let Spotify run N songs, embed them
    kick:     play the seed, then the kick track, let Spotify run N songs, embed them
Report each arm's mean distance from the seed and from the kick, and how many songs the arms share.
Headless, slow (every song is played to the end unless --skip-after is given), and honest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..ears import clap, previews
from ..player import spotify


@dataclass
class Arm:
    name: str
    plays: list[dict] = field(default_factory=list)      # {artist, title, uri, embedding}

    def vectors(self) -> np.ndarray | None:
        vs = [p["embedding"] for p in self.plays if p.get("embedding") is not None]
        return np.stack(vs) if vs else None


def _embed(store, embedder, t: spotify.Track):
    tr = store.track_by_uri(t.uri) or store.upsert_track(t.artist, t.name, album=t.album, spotify_uri=t.uri, duration_s=t.duration_s,
                                                          resolved_how="played")
    if not tr.preview_url:
        p = previews.lookup(tr.artist, tr.title)
        if p is None or not p.preview_url:
            return None
        tr = store.upsert_track(tr.artist, tr.title, itunes_id=p.itunes_id, preview_url=p.preview_url, duration_s=p.duration_s)
    return clap.embed_track(store, embedder, tr)


def _watch(store, embedder, n: int, *, skip_after: float | None, exclude: set[str], log, max_s: float) -> list[dict]:
    """Collect the next n distinct tracks Spotify plays on its own, embedding each."""
    out, last, t_end, started = [], None, time.time() + max_s, time.time()
    while len(out) < n and time.time() < t_end:
        try:
            t = spotify.now_playing()
        except spotify.PlayerError:
            t = None
        if t is not None and t.uri != last:
            last, started = t.uri, time.time()
            if t.uri not in exclude:
                out.append({"artist": t.artist, "title": t.name, "uri": t.uri, "embedding": _embed(store, embedder, t)})
                log(f"  {len(out)}/{n} {t.label}")
        if skip_after and t is not None and time.time() - started > skip_after and t.remaining_s > 5:
            spotify.next_track()
        time.sleep(2.0)
    return out


def run(store, embedder, seed_uri: str, kick_uri: str, *, n: int = 6, skip_after: float | None = None, log=print) -> dict:
    """Both arms, control first. Mutes Spotify for the duration and restores the volume after."""
    vol = spotify.volume()
    spotify.set_volume(0)
    try:
        per_song = (skip_after or 300) + 10
        log("control arm: seed, then Spotify alone")
        seed = spotify.play_and_confirm(seed_uri)
        seed_vec = _embed(store, embedder, seed)
        control = Arm("control", _watch(store, embedder, n, skip_after=skip_after, exclude={seed_uri}, log=log, max_s=per_song * (n + 1)))
        log("kick arm: seed, kick, then Spotify alone")
        spotify.play_and_confirm(seed_uri)
        time.sleep(skip_after or 30)
        kick = spotify.play_and_confirm(kick_uri)
        kick_vec = _embed(store, embedder, kick)
        kicked = Arm("kick", _watch(store, embedder, n, skip_after=skip_after, exclude={seed_uri, kick_uri}, log=log, max_s=per_song * (n + 1)))
    finally:
        spotify.set_volume(vol)

    def summary(arm: Arm) -> dict:
        V = arm.vectors()
        d_seed = float(np.mean(1.0 - V @ seed_vec)) if V is not None and seed_vec is not None else None
        d_kick = float(np.mean(1.0 - V @ kick_vec)) if V is not None and kick_vec is not None else None
        return {"n": len(arm.plays), "mean_from_seed": d_seed, "mean_from_kick": d_kick,
                "tracks": [f"{p['artist']} — {p['title']}" for p in arm.plays]}

    shared = {p["uri"] for p in control.plays} & {p["uri"] for p in kicked.plays}
    seed_kick = float(1.0 - seed_vec @ kick_vec) if seed_vec is not None and kick_vec is not None else None
    result = {"seed": seed.label, "kick": kick.label, "seed_to_kick": seed_kick, "control": summary(control), "kicked": summary(kicked),
              "shared_tracks": len(shared)}
    c, k = result["control"], result["kicked"]
    if c["mean_from_kick"] is not None and k["mean_from_kick"] is not None:
        result["kick_pulled_queue_toward_it"] = k["mean_from_kick"] < c["mean_from_kick"]
        result["kick_pushed_queue_from_seed"] = k["mean_from_seed"] > c["mean_from_seed"]
    return result
