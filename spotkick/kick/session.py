"""One listener, one Spotify, one kick at a time.

    observe()          poll the player; ingest what's playing into the store and the state; attribute it to the
                       active kick and update its verdict; keep the candidate pool warm; play follow-through songs
    kick(magnitude)    choose the prefetched candidate whose *measured* distance is nearest the wind-up, play it,
                       confirm the player has it, log the kick, start the follow-through for kick/boot

The Brain is only ever asked for names (brain.propose). Resolving, embedding, measuring, choosing, playing, and
judging are all done here, in the ruler's space, and every step lands in the store.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..brain import propose as PR
from ..brain import resolve as RS
from ..ears import clap, previews
from ..player import spotify
from . import bands as B

POOL_MAX_AGE_S = 60 * 60     # a pool built an hour ago is re-measured, not discarded
OFF_TARGET_REL = 0.5         # if the best candidate's rel is further than this from the target, rebuild before kicking
FOLLOW_END_S = 4.0           # play the next forced song when this much of the current one is left


@dataclass
class PoolItem:
    cand_id: int
    track: object              # store.Track, with spotify_uri
    embedding: np.ndarray
    reach: str
    direction: str
    why: str


@dataclass
class Pool:
    set_id: str
    for_track_id: int | None
    built_at: float
    items: list[PoolItem] = field(default_factory=list)


@dataclass
class ActiveKick:
    id: int
    pre: np.ndarray
    kick_vec: np.ndarray
    direction: str
    strength: str
    dose: int
    track_uri: str
    forced_uris: list[str] = field(default_factory=list)     # follow-through queue, in order
    forced_seen: set[str] = field(default_factory=set)        # every uri we forced, incl. the kick track
    n_since: int = 0
    followed: float = 0.0


class KickSession:
    def __init__(self, cfg, store, backend, embedder: clap.Embedder | None = None, *, player=spotify, searcher=None,
                 log=print, http=None):
        self.cfg, self.store, self.backend, self.player = cfg, store, backend, player
        self.embedder = embedder or clap.Embedder()
        self.searcher = searcher or getattr(backend, "search_uri", None)
        self.log, self.http = log, http
        self.state = B.ListenerState(alpha=cfg.alpha)
        self.pool: Pool | None = None
        self.active: ActiveKick | None = None
        self._pool_lock = threading.Lock()
        self._pool_thread: threading.Thread | None = None
        self._follow_thread: threading.Thread | None = None
        self._last_uri: str | None = None
        self._last_track = None
        self._max_pos = 0.0
        self._warm_state()
        self._restore_active()
        self._restore_pool()

    # ------------------------------------------------------------------ state
    def _warm_state(self) -> None:
        """Rebuild the listener state from the last plays in the store, so a restart doesn't start from nothing."""
        rows = [r for r in self.store.recent(30) if r["kind"] == "play"][:20]
        ids = [self.store.find_track(r["artist"], r["title"]).id for r in rows]
        vecs = self.store.embeddings(ids)
        for tid in reversed(ids):
            if tid in vecs:
                self.state.update(vecs[tid])

    def _restore_active(self, max_age_s: float = 60 * 60) -> None:
        """A kick from the last hour is still being judged, even across a restart (its follow-through queue is not)."""
        k = self.store.last_kick()
        if not k or time.time() - k["t"] > max_age_s or k["pre_state"] is None or k["kick_vec"] is None:
            return
        tr = self.store.track(k["track_id"])
        if tr is None or not tr.spotify_uri:
            return
        forced = {e["track_id"] for e in self.store.events(kinds=("play",), since=k["t"]) if e["source"] == "kick"}
        uris = {self.store.track(i).spotify_uri for i in forced if self.store.track(i)} | {tr.spotify_uri}
        self.active = ActiveKick(id=k["id"], pre=k["pre_state"], kick_vec=k["kick_vec"], direction=k["direction"] or "", strength=k["strength"],
                                 dose=k["dose"], track_uri=tr.spotify_uri, forced_seen=uris, n_since=k["n_since"] or 0, followed=k["followed"] or 0.0)
        cur = None
        try:
            cur = self.player.now_playing()
        except spotify.PlayerError:
            pass
        last = self.store.recent(1)
        last_uri = None
        if last:
            lt = self.store.find_track(last[0]["artist"], last[0]["title"])
            last_uri = lt.spotify_uri if lt else None
        if cur is not None and cur.uri == last_uri:
            self._last_uri = cur.uri  # the song playing now is the last one logged; don't count it twice

    def _restore_pool(self, max_age_s: float = POOL_MAX_AGE_S) -> None:
        """The latest candidate set survives a restart: embeddings are in the store, so no Brain call is needed."""
        rows = self.store.latest_candidate_set()
        rows = [r for r in rows if time.time() - r["t"] <= max_age_s]
        if not rows:
            return
        vecs = self.store.embeddings([r["track_id"] for r in rows])
        items = []
        for r in rows:
            tr = self.store.track(r["track_id"])
            if tr is not None and tr.spotify_uri and r["track_id"] in vecs:
                items.append(PoolItem(r["id"], tr, vecs[r["track_id"]], r["reach"] or "", r["direction"] or "", r["why"] or ""))
        if items:
            self.pool = Pool(set_id=rows[0]["set_id"], for_track_id=rows[0]["for_track_id"], built_at=rows[0]["t"], items=items)
            self.log(f"pool {self.pool.set_id} restored: {len(items)} picks")

    def _ctx(self) -> dict:
        step, far = self.state.scale()
        return {"typical_step": round(step, 4), "far": round(far, 4), "n_state": len(self.state.history),
                "kick_id": self.active.id if self.active else None, "dig": self.cfg.dig}

    # ---------------------------------------------------------------- observe
    def observe(self) -> dict:
        try:
            t = self.player.now_playing()
        except spotify.PlayerError as e:
            return {"error": str(e), "track": None, **self.snapshot()}
        if t is not None and t.uri != self._last_uri:
            self._close_previous()
            self._on_new_track(t)
            self._last_uri, self._last_track, self._max_pos = t.uri, t, 0.0
        elif t is not None:
            self._max_pos = max(self._max_pos, t.position_s)
            self._last_track = t
        self._maybe_follow_through(t)
        self._maybe_prefetch()
        return {"track": t, **self.snapshot()}

    def _close_previous(self) -> None:
        """When the track changes, judge the one that just ended: a skip is a signal the store should have."""
        prev = self._last_track
        if prev is None or prev.duration_s <= 0:
            return
        frac = self._max_pos / prev.duration_s
        if frac < 0.3:
            tr = self.store.track_by_uri(prev.uri)
            if tr is not None:
                self.store.add_event("skip", tr.id, "spotify", skip_at_s=self._max_pos, completion=frac, ctx=self._ctx())

    def _on_new_track(self, t) -> None:
        forced = self.active is not None and t.uri in self.active.forced_seen
        source = "kick" if forced else "spotify"
        tr = self.store.track_by_uri(t.uri) or self.store.upsert_track(t.artist, t.name, album=t.album, spotify_uri=t.uri,
                                                                          duration_s=t.duration_s, resolved_how="played")
        vec = self._embed(tr)
        if vec is not None:
            self.state.update(vec)
        self.store.add_event("play", tr.id, source, popularity=t.popularity, kick_id=self.active.id if self.active else None,
                             ctx=self._ctx())
        if self.active and not forced and vec is not None:
            k = self.active
            k.n_since += 1
            k.followed = B.followed(k.pre, k.kick_vec, self.state.vector)
            v = B.verdict(k.followed, k.n_since)
            self.store.update_kick(k.id, followed=k.followed, verdict=v, verdict_at=time.time(), n_since=k.n_since)
            self.log(f"since kick #{k.id}: {t.artist} — {t.name} · followed {k.followed:.2f} → {v}")
        if forced and t.uri == self.active.track_uri and t.popularity is not None:
            self.store.update_kick(self.active.id, popularity=t.popularity)

    def _embed(self, tr) -> np.ndarray | None:
        v = self.store.embedding(tr.id)
        if v is not None:
            return v
        if not tr.preview_url:
            p = previews.lookup(tr.artist, tr.title, session=self.http)
            if p is None or not p.preview_url:
                self.log(f"no preview for {tr.label}; not in the state")
                return None
            tr = self.store.upsert_track(tr.artist, tr.title, album=p.album, itunes_id=p.itunes_id, preview_url=p.preview_url,
                                         duration_s=p.duration_s)
        try:
            return clap.embed_track(self.store, self.embedder, tr)
        except Exception as e:  # noqa: BLE001 — network, ffmpeg; the song just stays out of the state
            self.log(f"embed failed for {tr.label}: {e}")
            return None

    # --------------------------------------------------------------- prefetch
    def _current_track_id(self) -> int | None:
        cur = self.store.track_by_uri(self._last_uri) if self._last_uri else None
        return cur.id if cur else None

    def _maybe_prefetch(self) -> None:
        if self._pool_thread and self._pool_thread.is_alive():
            return
        cur_id = self._current_track_id()
        stale = self.pool is None or not self.pool.items or self._off_track(cur_id)
        if stale and self.state.vector is not None:
            self._pool_thread = threading.Thread(target=self._build_pool, args=(cur_id,), daemon=True)
            self._pool_thread.start()

    def _off_track(self, cur_id: int | None, min_age_s: float = 120.0) -> bool:
        """The pool was built while something else was playing and is old enough to be worth replacing."""
        return self.pool is not None and self.pool.for_track_id != cur_id and time.time() - self.pool.built_at > min_age_s

    def _build_pool(self, for_track_id: int | None, *, n: int | None = None, direction_hint: str | None = None) -> Pool:
        t0 = time.time()
        set_id = uuid.uuid4().hex[:12]
        cands = PR.propose(self.backend, self.store, n=n or self.cfg.n_candidates, dig=self.cfg.dig, direction_hint=direction_hint,
                           log=self.log)
        ids = self.store.add_candidates(set_id, [c.as_row() for c in cands], for_track_id=for_track_id,
                                        purpose="follow" if direction_hint else "pool")
        fresh = [(cid, c) for cid, c in zip(ids, cands) if not c.rejected_reason]
        t_llm = time.time() - t0
        with ThreadPoolExecutor(max_workers=6) as ex:
            items = [it for it in ex.map(lambda p: self._materialize(*p), fresh) if it is not None]
        pool = Pool(set_id=set_id, for_track_id=for_track_id, built_at=time.time(), items=items)
        if direction_hint is None:
            with self._pool_lock:
                self.pool = pool
        self.log(f"pool {set_id}: {len(items)}/{len(fresh)} usable · brain {t_llm:.0f}s · total {time.time()-t0:.0f}s")
        return pool

    def _materialize(self, cand_id: int, c: PR.Candidate) -> PoolItem | None:
        """Names → a playable, measurable thing: preview (iTunes), embedding (ruler), URI (resolver). Logged either way."""
        p = previews.lookup(c.artist, c.title, session=self.http)
        if p is None or not p.preview_url:
            self.store.update_candidate(cand_id, rejected_reason="no preview")
            return None
        r = RS.resolve(c.artist, c.title, c.spotify_uri, searcher=self.searcher, log=self.log)
        if r is None:
            self.store.update_candidate(cand_id, rejected_reason="no trusted uri")
            return None
        tr = self.store.upsert_track(p.artist, p.title, album=p.album, spotify_uri=r.uri, itunes_id=p.itunes_id, preview_url=p.preview_url,
                                     duration_s=p.duration_s, resolved_how=r.how)
        try:
            vec = clap.embed_track(self.store, self.embedder, tr)
        except Exception as e:  # noqa: BLE001 — logged as a rejected candidate
            self.store.update_candidate(cand_id, rejected_reason=f"embed failed: {e}")
            return None
        self.store.update_candidate(cand_id, track_id=tr.id)
        return PoolItem(cand_id, tr, vec, c.reach, c.direction, c.why)

    def ready(self) -> int:
        with self._pool_lock:
            return len(self.pool.items) if self.pool else 0

    def wait_for_pool(self, timeout_s: float = 120.0) -> int:
        """Block until candidates are available (the kick path when the prefetch hasn't finished)."""
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            if self.ready():
                return self.ready()
            if not (self._pool_thread and self._pool_thread.is_alive()):
                self._pool_thread = threading.Thread(target=self._build_pool, args=(None,), daemon=True)
                self._pool_thread.start()
            time.sleep(0.5)
        return self.ready()

    # ------------------------------------------------------------------- kick
    def _measure(self, items: list[PoolItem]) -> list[B.Measured]:
        measured = B.measure(self.state, [it.embedding for it in items])
        for it, m in zip(items, measured):
            self.store.update_candidate(it.cand_id, distance=m.distance, rel=m.rel, band=m.band)
        return measured

    def kick(self, magnitude: float) -> dict:
        if self.state.vector is None:
            raise RuntimeError("nothing has played yet; play a song first so there is somewhere to kick from")
        if not self.ready():
            self.wait_for_pool()
        with self._pool_lock:
            pool, items = self.pool, list(self.pool.items) if self.pool else []
        if not items:
            raise RuntimeError("no playable candidates; the brain or the resolver came up empty")
        strength, target, dose = B.strength_for(magnitude), B.target_for(magnitude), B.dose_for(magnitude)
        measured = self._measure(items)
        m = B.choose(measured, target)
        if abs(m.rel - target) > OFF_TARGET_REL:
            # nothing in the pool lands anywhere near the wind-up (the pool is left over from elsewhere, or only one reach
            # survived resolution): ask for a fresh spread rather than lie with the least-wrong leftover
            self.log(f"pool off target: best rel {m.rel:.2f} vs {target:.2f} among {len(items)} — rebuilding")
            fresh = self._build_pool(self._current_track_id())
            if fresh.items:
                pool, items = fresh, list(fresh.items)
                measured = self._measure(items)
                m = B.choose(measured, target)
        it = items[m.index]
        pre = self.state.vector.copy()
        played = self.player.play_and_confirm(it.track.spotify_uri)
        kick_id = self.store.add_kick(strength=strength, magnitude=float(magnitude), target_rel=target, direction=it.direction, why=it.why,
                                      track_id=it.track.id, distance=m.distance, rel=m.rel, band=m.band, dose=dose, pre_state=pre,
                                      kick_vec=it.embedding, popularity=played.popularity)
        self.store.update_candidate(it.cand_id, chosen=1, kick_id=kick_id)
        self.store.add_event("kick", it.track.id, "kick", kick_id=kick_id, popularity=played.popularity, ctx=self._ctx())
        self.active = ActiveKick(id=kick_id, pre=pre, kick_vec=it.embedding, direction=it.direction, strength=strength, dose=dose,
                                 track_uri=it.track.spotify_uri, forced_seen={it.track.spotify_uri})
        # the kick track is playing now: ingest it here so the log has it even if nobody observes again
        self._close_previous()
        self.state.update(it.embedding)
        self.store.add_event("play", it.track.id, "kick", popularity=played.popularity, kick_id=kick_id, ctx=self._ctx())
        self._last_uri, self._last_track, self._max_pos = it.track.spotify_uri, played, 0.0
        with self._pool_lock:
            if self.pool is pool:
                self.pool.items = [x for x in self.pool.items if x.cand_id != it.cand_id]
        self.log(f"kick #{kick_id} {strength} (target {target:.2f}) → {it.track.label} · measured {m.distance:.3f} rel {m.rel:.2f} "
                 f"[{m.band}] · {it.direction}")
        if dose > 1:
            self._follow_thread = threading.Thread(target=self._fetch_follow_through, args=(self.active, dose - 1), daemon=True)
            self._follow_thread.start()
        return {"kick_id": kick_id, "strength": strength, "target_rel": target, "dose": dose, "track": it.track, "direction": it.direction,
                "why": it.why, "distance": m.distance, "rel": m.rel, "band": m.band, "acceptance": m.acceptance,
                "candidates": [{"artist": x.track.artist, "title": x.track.title, "reach": x.reach, "distance": mm.distance, "rel": mm.rel,
                                "band": mm.band, "chosen": x is it} for x, mm in zip(items, measured)]}

    # --------------------------------------------------------- follow-through
    def _fetch_follow_through(self, k: ActiveKick, n: int) -> None:
        """While the kick song plays, get n more songs in its direction, ordered; they're played at song end."""
        try:
            pool = self._build_pool(None, n=n, direction_hint=k.direction)
            k.forced_uris = [it.track.spotify_uri for it in pool.items[:n]]
            k.forced_seen.update(k.forced_uris)
            self.log(f"follow-through for kick #{k.id}: {len(k.forced_uris)} songs ready")
        except Exception as e:  # noqa: BLE001 — a failed follow-through must not kill the observe loop
            self.log(f"follow-through failed: {e}")

    def _maybe_follow_through(self, t) -> None:
        k = self.active
        if k is None or not k.forced_uris or t is None:
            return
        ending = t.uri in k.forced_seen and t.duration_s > 0 and t.remaining_s <= FOLLOW_END_S
        spotify_moved_on = t.uri not in k.forced_seen and t.position_s < 20
        if ending or spotify_moved_on:
            uri = k.forced_uris.pop(0)
            try:
                self.player.play(uri)
                self.log(f"follow-through: {uri} ({len(k.forced_uris)} left)")
            except spotify.PlayerError as e:
                self.log(f"follow-through play failed: {e}")

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        step, far = self.state.scale()
        k = self.active
        last = self.store.kick(k.id) if k else None
        return {"state": {"n": len(self.state.history), "typical_step": step, "far": far},
                "pool": {"ready": self.ready(), "building": bool(self._pool_thread and self._pool_thread.is_alive())},
                "kick": None if not last else {"id": k.id, "strength": k.strength, "direction": k.direction, "dose": k.dose,
                                               "track": self.store.track(last["track_id"]), "distance": last["distance"], "rel": last["rel"],
                                               "band": last["band"], "popularity": last["popularity"], "n_since": k.n_since,
                                               "followed": k.followed, "verdict": B.verdict(k.followed, k.n_since),
                                               "forced_left": len(k.forced_uris)}}
