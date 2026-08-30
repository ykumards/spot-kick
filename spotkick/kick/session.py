"""The kick loop for one listener and one Spotify instance.

``observe()`` polls the player, records what is playing in the store and the listener state, judges the active
kick by it, and keeps the candidate pool stocked. ``kick(magnitude)`` plays the prefetched candidate whose
measured distance is nearest the wind-up, confirms playback, and records the kick.

The brain is only asked for names (``brain.propose``); resolving, embedding, measuring, choosing, playing and
judging happen here, and every step is written to the store.

Threading: ``observe()`` runs on the caller's thread; ``build_pool()`` usually runs on a daemon thread started by
``maybe_prefetch()``. ``self.pool`` is only replaced or trimmed under ``_pool_lock``; everything else is
single-threaded.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..brain import propose, resolve
from ..brain.llm import BrainError
from ..ears import clap, previews
from ..memory.store import Store, Track
from ..player import spotify
from ..player.spotify_api import NOT_CONFIGURED_MESSAGE, SpotifyAPI
from . import bands

if TYPE_CHECKING:
    import requests

    from ..brain.llm import Backend
    from ..config import Config

LIBRARY_SET_ID = "library"     # the set id of a pool assembled from stored candidates rather than one brain call
ACTIVE_KICK_MAX_AGE_S = 60 * 60  # a kick older than this is no longer being judged after a restart
POOL_OFF_TRACK_MIN_AGE_S = 120.0  # a pool built for another song is only replaced once it is this old
OFF_TARGET_WARN_REL = 0.5      # best candidate's rel further than this from the target is logged, not waited on
STEPS_SHOWN = 40               # consecutive-song distances the stats screen plots
BAND_TOP_UP_N = 4              # candidates asked for when one band of the pool has run dry
BAND_RETRY_N = 6               # ... and on every retry, with the misses fed back
# An empty band is asked for again and again, each time telling the brain where its last picks measured. The
# delays only pace the brain's quota: two immediate corrections, then a minute, then five, then ten between tries.
BAND_RETRY_DELAYS_S = (0.0, 0.0, 60.0, 300.0, 600.0)
BAND_MISSES_KEPT = 12          # measured misses remembered per band and fed back to the brain
LEAN_CAP_TRIES = 2             # misses in a row under a lean before we say the lean itself caps the reach
REACH_FOR_BAND = {"tap": "near", "kick": "adjacent", "boot": "far"}
SKIP_COMPLETION = 0.3          # leaving a song before this fraction of it counts as a skip
WARM_STATE_RECENT_ROWS = 30    # store rows scanned to rebuild the listener state ...
WARM_STATE_PLAYS = 20          # ... and how many plays of those are replayed into it
MATERIALIZE_WORKERS = 6        # candidates resolved/embedded in parallel
POOL_WAIT_POLL_S = 0.5
SET_ID_LENGTH = 12

NO_STATE_MESSAGE = "nothing has played yet; play a song first so there is somewhere to kick from"
NO_CANDIDATES_MESSAGE = "no playable candidates; the brain or Spotify's search came up empty"
NOT_PLAYING_MESSAGE = ("Spotify reports nothing playing on this Mac; press play there first (right after Spotify "
                       "launches it ignores play requests for a while)")

Logger = Callable[[str], None]


class Player(Protocol):
    """The player interface the session needs; the ``spotify`` module and the test fake both satisfy it."""

    def now_playing(self) -> spotify.Track | None: ...

    def play_and_confirm(self, uri: str, *, timeout_s: float = ...) -> spotify.Track: ...


@dataclass
class PoolItem:
    cand_id: int
    track: Track
    uri: str                   # the track's Spotify URI; every pool item is playable by construction
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
    """The kick being judged: the inputs to ``followed`` and the URI that identifies the kicked track."""
    id: int
    pre: np.ndarray
    kick_vec: np.ndarray
    track_uri: str
    n_since: int = 0
    followed: float = 0.0


def without_repeated_tracks(items: list[PoolItem]) -> list[PoolItem]:
    """Return the items with duplicate tracks removed, keeping the first of each.

    The same song can be proposed in a build and a later top-up, because the brain is only told about songs
    already played.
    """
    seen_tracks: set[int] = set()
    unique = []
    for item in items:
        if item.track.id in seen_tracks:
            continue
        seen_tracks.add(item.track.id)
        unique.append(item)
    return unique


class KickSession:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        backend: Backend,
        embedder: clap.Embedder | None = None,
        *,
        player: Player = spotify,
        api: SpotifyAPI | None = None,
        log: Logger = print,
        http: requests.Session | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.backend = backend
        self.player = player
        self.embedder = embedder or clap.Embedder()
        self.api = api or SpotifyAPI.from_config(cfg)
        self.log = log
        self.http = http
        self._warned_no_credentials = False
        self.state = bands.ListenerState(alpha=cfg.alpha)
        self.pool: Pool | None = None
        self.active: ActiveKick | None = None
        self._pool_lock = threading.Lock()
        self._pool_thread: threading.Thread | None = None
        self._band_asked_at: dict[str, float] = {}
        self._band_attempts: dict[str, int] = {}       # top-ups that came back with nothing in the band, in a row
        self._band_misses: dict[str, list[dict]] = {}  # what those top-ups produced and where each measured
        self._pool_generation = 0     # bumped by invalidate_pool, so a build started under old settings is dropped
        self._building_reach: str | None = None   # the reach a running top-up asks for; None when idle or a full build
        self._last_uri: str | None = None
        self._last_track: spotify.Track | None = None
        self._max_pos = 0.0
        self.warm_state()
        self.skip_already_logged_song()
        self.restore_active_kick()
        self.restore_pool()

    # ------------------------------------------------------------------ state
    def warm_state(self) -> None:
        """Rebuild the listener state from the most recent plays in the store."""
        recent = self.store.recent(WARM_STATE_RECENT_ROWS)
        plays = [row for row in recent if row["kind"] == "play"][:WARM_STATE_PLAYS]
        replay: list[tuple[int, bool]] = []
        for row in plays:
            track = self.store.find_track(row["artist"], row["title"])
            if track is not None:
                replay.append((track.id, row["source"] != "kick"))
        embeddings = self.store.embeddings([track_id for track_id, _ in replay])
        for track_id, spotify_chose_it in reversed(replay):
            if track_id in embeddings:
                self.state.update(embeddings[track_id], counts_for_scale=spotify_chose_it)

    def skip_already_logged_song(self) -> None:
        """Treat the track playing at startup as already observed when it is the last one logged.

        Logging it again would add a zero-distance play and collapse the listener's scale.
        """
        playing = self.playing_now_or_none()
        if playing is not None and playing.uri == self.last_logged_uri():
            self._last_uri = playing.uri
            self._last_track = playing
            self._max_pos = playing.position_s

    def restore_active_kick(self, max_age_s: float = ACTIVE_KICK_MAX_AGE_S) -> None:
        """Restore the active kick from the store when the last kick is recent enough to still be judged."""
        kick = self.store.last_kick()
        if not kick or time.time() - kick["t"] > max_age_s:
            return
        if kick["pre_state"] is None or kick["kick_vec"] is None:
            return
        track = self.store.track(kick["track_id"])
        if track is None or not track.spotify_uri:
            return
        self.active = ActiveKick(id=kick["id"], pre=kick["pre_state"], kick_vec=kick["kick_vec"],
                                 track_uri=track.spotify_uri, n_since=kick["n_since"] or 0,
                                 followed=kick["followed"] or 0.0)

    def playing_now_or_none(self) -> spotify.Track | None:
        try:
            return self.player.now_playing()
        except spotify.PlayerError:
            return None

    def last_logged_uri(self) -> str | None:
        """Return the Spotify URI of the most recently logged track, or None."""
        recent = self.store.recent(1)
        if not recent:
            return None
        track = self.store.find_track(recent[0]["artist"], recent[0]["title"])
        if track is None:
            return None
        return track.spotify_uri

    def restore_pool(self) -> None:
        """Restore the pool from the library, so a restart needs no brain call to have candidates."""
        pool = self.pool_from_library(for_track_id=None)
        if pool is None:
            return
        self.pool = pool
        self.log(f"pool {pool.set_id} restored from the library: {len(pool.items)} picks · {self.coverage_line()}")

    def pool_from_library(self, for_track_id: int | None) -> Pool | None:
        """Build a pool from the current lean's library of resolved, embedded candidates; None when it is empty."""
        rows = self.store.library_candidates(self.cfg.lean)
        if not rows:
            return None
        embeddings = self.store.embeddings([row["track_id"] for row in rows])
        items = []
        for row in rows:
            track = self.store.track(row["track_id"])
            if track is None or not track.spotify_uri or row["track_id"] not in embeddings:
                continue
            items.append(PoolItem(row["id"], track, track.spotify_uri, embeddings[row["track_id"]],
                                  row["reach"] or "", row["direction"] or "", row["why"] or ""))
        if not items:
            return None
        return Pool(set_id=LIBRARY_SET_ID, for_track_id=for_track_id, built_at=time.time(), items=items)

    def refill_from_library(self, for_track_id: int | None) -> bool:
        """Replace a missing or stale pool with the library. Returns False when the library is empty."""
        pool = self.pool_from_library(for_track_id)
        if pool is None:
            return False
        with self._pool_lock:
            self.pool = pool
        self.log(f"pool refilled from the library: {len(pool.items)} picks · {self.coverage_line()}")
        return True

    def _active_kick_id(self) -> int | None:
        if self.active is None:
            return None
        return self.active.id

    # ---------------------------------------------------------------- observe
    def observe(self) -> dict:
        try:
            playing = self.player.now_playing()
        except spotify.PlayerError as error:
            return {"error": str(error), "track": None, **self.snapshot()}
        if playing is not None and playing.uri != self._last_uri:
            self.judge_previous_track()
            self.ingest_track(playing)
            self._last_uri = playing.uri
            self._last_track = playing
            self._max_pos = 0.0
        elif playing is not None:
            self._max_pos = max(self._max_pos, playing.position_s)
            self._last_track = playing
        self.maybe_prefetch()
        return {"track": playing, **self.snapshot()}

    def judge_previous_track(self) -> None:
        """Record a skip event for the track that just ended, if the listener left it early."""
        previous = self._last_track
        if previous is None or previous.duration_s <= 0:
            return
        completion = self._max_pos / previous.duration_s
        if completion >= SKIP_COMPLETION:
            return
        track = self.store.track_by_uri(previous.uri)
        if track is not None:
            self.store.add_event("skip", track.id, "spotify", skip_at_s=self._max_pos, completion=completion)

    def ingest_track(self, playing: spotify.Track) -> None:
        """Record a newly playing track in the store and the state, and judge the active kick by it."""
        kick = self.active
        is_kick_track = kick is not None and playing.uri == kick.track_uri
        source = "kick" if is_kick_track else "spotify"
        track = self.store.track_by_uri(playing.uri)
        if track is None:
            track = self.store.upsert_track(playing.artist, playing.name, album=playing.album, spotify_uri=playing.uri,
                                            duration_s=playing.duration_s)
        embedding = self.embedding_for(track)
        if embedding is not None:
            self.state.update(embedding, counts_for_scale=not is_kick_track)
        self.store.add_event("play", track.id, source, popularity=playing.popularity, kick_id=self._active_kick_id())
        if kick is None:
            return
        if is_kick_track:
            if playing.popularity is not None:
                self.store.update_kick(kick.id, popularity=playing.popularity)
        elif embedding is not None:
            self.judge_active_kick(kick, playing)

    def judge_active_kick(self, kick: ActiveKick, playing: spotify.Track) -> None:
        """Update the active kick's ``followed`` and verdict for one more recommender-chosen song.

        The verdict is frozen after SONGS_TO_JUDGE songs: the state is an exponentially weighted mean, so measuring
        further would report the drift of the whole session rather than the response to this kick.
        """
        if kick.n_since >= bands.SONGS_TO_JUDGE:
            return
        now = self.state.vector
        if now is None:
            return  # nothing has reached the state yet, so there is no movement to measure
        kick.n_since += 1
        kick.followed = bands.followed(kick.pre, kick.kick_vec, now)
        verdict = bands.verdict(kick.followed, kick.n_since)
        self.store.update_kick(kick.id, followed=kick.followed, verdict=verdict, verdict_at=time.time(),
                               n_since=kick.n_since)
        self.log(f"since kick #{kick.id}: {playing.artist} — {playing.name} · followed {kick.followed:.2f} → {verdict}")

    def embedding_for(self, track: Track) -> np.ndarray | None:
        """Return the track's embedding from the store, computing it from a preview when necessary. None on failure."""
        stored = self.store.embedding(track.id)
        if stored is not None:
            return stored
        if not track.preview_url:
            preview = previews.lookup(track.artist, track.title, session=self.http)
            if preview is None or not preview.preview_url:
                self.log(f"no preview for {track.label}; not in the state")
                return None
            track = self.store.upsert_track(track.artist, track.title, album=preview.album,
                                            preview_url=preview.preview_url, duration_s=preview.duration_s)
        try:
            return clap.embed_track(self.store, self.embedder, track)
        except Exception as error:  # noqa: BLE001 — network, afconvert; the song just stays out of the state
            self.log(f"embed failed for {track.label}: {error}")
            return None

    # --------------------------------------------------------------- prefetch
    def _current_track_id(self) -> int | None:
        if not self._last_uri:
            return None
        track = self.store.track_by_uri(self._last_uri)
        if track is None:
            return None
        return track.id

    def _pool_building(self) -> bool:
        return bool(self._pool_thread and self._pool_thread.is_alive())

    def _start_pool_thread(self, for_track_id: int | None, reach: str | None = None) -> None:
        self._pool_thread = threading.Thread(target=self._build_pool_quietly, args=(for_track_id, reach), daemon=True)
        self._pool_thread.start()

    def _build_pool_quietly(self, for_track_id: int | None, reach: str | None = None) -> None:
        """Run a build or top-up on the background thread, logging any failure and leaving the existing pool intact."""
        self._building_reach = reach
        try:
            if reach is None:
                self.build_pool(for_track_id)
            else:
                self.top_up_pool(for_track_id, reach)
        except BrainError as error:
            self.log(f"prefetch skipped: {error}")
        except Exception as error:  # noqa: BLE001 — a daemon thread must not die with a traceback on the log
            self.log(f"prefetch failed: {type(error).__name__}: {error}")
        finally:
            self._building_reach = None

    def maybe_prefetch(self) -> None:
        """Start a background build when there is no usable pool, or a top-up for the first empty band."""
        if self._pool_building() or self.state.vector is None:
            return
        if not self.api.configured:
            # Without credentials nothing can be resolved, so asking the brain would only spend its quota.
            if not self._warned_no_credentials:
                self.log(NOT_CONFIGURED_MESSAGE)
                self._warned_no_credentials = True
            return
        current_id = self._current_track_id()
        stale = self.pool is None or not self.pool.items or self._pool_is_off_track(current_id)
        if stale and not self.refill_from_library(current_id):
            self._start_pool_thread(current_id)
            return
        band = self.first_empty_band()
        if band is not None:
            self._start_pool_thread(current_id, reach=REACH_FOR_BAND[band])

    def pool_bands(self) -> dict[str, int]:
        """Return how many pool items currently measure into each band."""
        counts = {band: 0 for band in bands.STRENGTHS}
        with self._pool_lock:
            items = list(self.pool.items) if self.pool is not None else []
        for item in items:
            counts[self.state.band_for(self.state.distance(item.embedding))] += 1
        return counts

    def bench(self) -> dict[str, dict]:
        """Return the pool per band: its picks nearest first, the one a kick of that strength would play, and the
        band's state.
        """
        with self._pool_lock:
            items = list(self.pool.items) if self.pool is not None else []
        picks: dict[str, list[dict]] = {band: [] for band in bands.STRENGTHS}
        for item in items:
            distance = self.state.distance(item.embedding)
            rel = self.state.rel(distance)
            band = self.state.band_for(distance)
            picks[band].append({"cand_id": item.cand_id, "artist": item.track.artist, "title": item.track.title,
                                "rel": rel, "distance": distance, "direction": item.direction, "why": item.why})
        building = self._pool_building()
        now = time.time()
        result = {}
        for band in bands.STRENGTHS:
            ordered = sorted(picks[band], key=lambda pick: pick["distance"])
            would_play = min(picks[band], key=lambda pick: abs(pick["rel"] - bands.TARGET_REL[band]), default=None)
            result[band] = {"picks": ordered, "would_play": would_play,
                            "state": self.band_state(band, len(ordered), building, now)}
        return result

    def band_state(self, band: str, count: int, building: bool, now: float) -> str:
        """Return the band's state: building, ready, capped, resting or empty.

        'capped' means the lean confines the reach: after LEAN_CAP_TRIES top-ups nothing inside the lean measured
        this far.
        """
        if building and (self._building_reach is None or self._building_reach == REACH_FOR_BAND[band]):
            return "building"
        if count > 0:
            return "ready"
        if self.cfg.lean and self._band_attempts.get(band, 0) >= LEAN_CAP_TRIES:
            return "capped"
        if self.band_is_waiting(band, now):
            return "resting"
        return "empty"

    def coverage_line(self) -> str:
        counts = self.pool_bands()
        return " ".join(f"{band} {count}" for band, count in counts.items())

    def band_retry_delay(self, band: str) -> float:
        """Return how long an empty band waits before another top-up, given the number of misses so far."""
        attempts = self._band_attempts.get(band, 0)
        return BAND_RETRY_DELAYS_S[min(attempts, len(BAND_RETRY_DELAYS_S) - 1)]

    def band_is_waiting(self, band: str, now: float) -> bool:
        return now - self._band_asked_at.get(band, 0.0) < self.band_retry_delay(band)

    def first_empty_band(self) -> str | None:
        """Return the empty band due for a top-up, least-tried first, or None.

        Every band gets a first attempt before any gets a second; the delay between attempts paces the brain's
        quota.
        """
        now = time.time()
        due = [band for band, count in self.pool_bands().items() if count == 0 and not self.band_is_waiting(band, now)]
        if not due:
            return None
        return min(due, key=lambda band: self._band_attempts.get(band, 0))

    def _pool_is_off_track(self, current_id: int | None, min_age_s: float = POOL_OFF_TRACK_MIN_AGE_S) -> bool:
        """Return True when the pool was built for another track and is older than ``min_age_s``."""
        if self.pool is None or self.pool.for_track_id == current_id:
            return False
        return time.time() - self.pool.built_at > min_age_s

    def build_pool(
        self,
        for_track_id: int | None,
        *,
        n: int | None = None,
        reach: str | None = None,
        misses: list[dict] | None = None,
    ) -> Pool:
        """Ask the brain for candidates and materialise them in parallel.

        Without ``reach`` the result becomes ``self.pool``. With ``reach`` it is one band's worth, returned for
        ``top_up_pool`` to merge, with the band's earlier misses included in the prompt.
        """
        started = time.time()
        generation = self._pool_generation
        set_id = uuid.uuid4().hex[:SET_ID_LENGTH]
        candidates = propose.propose(self.backend, self.store, n=n or self.cfg.n_candidates, reach=reach,
                                     lean=self.cfg.lean or None, misses=misses, log=self.log)
        candidate_ids = self.store.add_candidates(set_id, [candidate.as_row() for candidate in candidates],
                                                  for_track_id=for_track_id, lean=self.cfg.lean)
        fresh_ids = []
        fresh_candidates = []
        for candidate_id, candidate in zip(candidate_ids, candidates):
            if not candidate.rejected_reason:
                fresh_ids.append(candidate_id)
                fresh_candidates.append(candidate)
        brain_seconds = time.time() - started
        with ThreadPoolExecutor(max_workers=MATERIALIZE_WORKERS) as executor:
            materialized = executor.map(self.materialize_candidate, fresh_ids, fresh_candidates)
            items = [item for item in materialized if item is not None]
        pool = Pool(set_id=set_id, for_track_id=for_track_id, built_at=time.time(), items=items)
        if self.is_stale_generation(generation):
            self.log(f"pool {set_id} discarded: settings changed while it was being built")
            return Pool(set_id=set_id, for_track_id=for_track_id, built_at=pool.built_at, items=[])
        if reach is None:
            with self._pool_lock:
                self.pool = pool
        total_seconds = time.time() - started
        timing = f"brain {brain_seconds:.0f}s · total {total_seconds:.0f}s"
        what = f"pool {set_id}" if reach is None else f"top-up {reach} {set_id}"
        self.log(f"{what}: {len(items)}/{len(fresh_ids)} usable · {timing}")
        return pool

    def top_up_pool(self, for_track_id: int | None, reach: str) -> None:
        """Fetch one band's worth of candidates and merge them into the pool.

        A top-up that leaves the band empty is recorded as a miss, with where each pick measured, for the next
        attempt's prompt.
        """
        band = next(band for band, band_reach in REACH_FOR_BAND.items() if band_reach == reach)
        self._band_asked_at[band] = time.time()
        attempts = self._band_attempts.get(band, 0)
        n = BAND_RETRY_N if attempts else BAND_TOP_UP_N
        fresh = self.build_pool(for_track_id, n=n, reach=reach, misses=self._band_misses.get(band))
        if fresh.items:
            self.merge_into_pool(fresh)
        self.record_band_outcome(band, fresh)
        self.log(f"pool now {self.coverage_line()}")

    def record_band_outcome(self, band: str, fresh: Pool) -> None:
        """Record whether the top-up filled its band; on a miss, count it and remember where the picks measured."""
        if self.pool_bands()[band] > 0:
            self._band_attempts[band] = 0
            self._band_misses.pop(band, None)
            return
        self._band_attempts[band] = self._band_attempts.get(band, 0) + 1
        misses = self._band_misses.setdefault(band, [])
        for item in fresh.items:
            landed = self.state.band_for(self.state.distance(item.embedding))
            misses.append({"artist": item.track.artist, "title": item.track.title, "band": landed})
        del misses[:-BAND_MISSES_KEPT]
        attempt = self._band_attempts[band]
        self.log(f"{band} still empty after try {attempt}: {len(fresh.items)} picks landed elsewhere · "
                 f"next try in {self.band_retry_delay(band):.0f}s with them fed back")

    def merge_into_pool(self, fresh: Pool) -> None:
        """Add a set's items to the pool, or make them the pool when there is none."""
        with self._pool_lock:
            if self.pool is None:
                self.pool = fresh
                return
            self.pool.items = without_repeated_tracks(self.pool.items + fresh.items)

    def materialize_candidate(self, cand_id: int, candidate: propose.Candidate) -> PoolItem | None:
        """Turn a named candidate into a PoolItem: preview, Spotify URI, embedding.

        Returns None when any step fails; the reason is recorded on the candidate row.
        """
        preview = previews.lookup(candidate.artist, candidate.title, session=self.http)
        if preview is None or not preview.preview_url:
            self.store.update_candidate(cand_id, rejected_reason="no preview")
            return None
        resolved = resolve.resolve(candidate.artist, candidate.title, self.api, log=self.log)
        if resolved is None:
            self.store.update_candidate(cand_id, rejected_reason="not on spotify")
            return None
        track = self.store.upsert_track(preview.artist, preview.title, album=preview.album, spotify_uri=resolved.uri,
                                        preview_url=preview.preview_url, duration_s=preview.duration_s)
        if not track.spotify_uri:
            # The upsert attaches the resolved URI to a track that had none, so this only guards the type.
            self.store.update_candidate(cand_id, rejected_reason="not on spotify")
            return None
        try:
            embedding = clap.embed_track(self.store, self.embedder, track)
        except Exception as error:  # noqa: BLE001 — logged as a rejected candidate
            self.store.update_candidate(cand_id, rejected_reason=f"embed failed: {error}")
            return None
        if embedding is None:
            self.store.update_candidate(cand_id, rejected_reason="no preview")
            return None
        self.store.update_candidate(cand_id, track_id=track.id)
        return PoolItem(cand_id, track, track.spotify_uri, embedding, candidate.reach, candidate.direction,
                        candidate.why)

    def ready(self) -> int:
        with self._pool_lock:
            if self.pool is None:
                return 0
            return len(self.pool.items)

    def set_brain(self, backend: Backend) -> None:
        """Switch the brain backend and discard its predecessor's pool."""
        self.backend = backend
        self.invalidate_pool()

    def invalidate_pool(self) -> None:
        """Discard the pool and the band cooldowns after a setting changes; the next observation rebuilds."""
        with self._pool_lock:
            self.pool = None
            self._band_asked_at.clear()
            self._band_attempts.clear()
            self._band_misses.clear()
            self._pool_generation += 1

    def is_stale_generation(self, generation: int) -> bool:
        """Return True when the pool was invalidated after the build with this generation started."""
        with self._pool_lock:
            return generation != self._pool_generation

    def wait_for_pool(self, timeout_s: float = 120.0) -> int:
        """Block until the pool has candidates or ``timeout_s`` elapses, starting a build if none is running."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.ready():
                return self.ready()
            if not self._pool_building():
                self._start_pool_thread(None)
            time.sleep(POOL_WAIT_POLL_S)
        return self.ready()

    # ------------------------------------------------------------------- kick
    def measure_pool(self, items: list[PoolItem]) -> list[bands.Measured]:
        """Measure every item against the current state and record the numbers on the candidate rows."""
        measured = bands.measure(self.state, [item.embedding for item in items])
        for item, measurement in zip(items, measured):
            self.store.update_candidate(item.cand_id, distance=measurement.distance, rel=measurement.rel,
                                        band=measurement.band)
        return measured

    def measure_and_choose(self, items: list[PoolItem], target: float) -> tuple[list[bands.Measured], bands.Measured]:
        """Measure the items and choose the one nearest ``target``. Raises RuntimeError for an empty pool."""
        measured = self.measure_pool(items)
        best = bands.choose(measured, target)
        if best is None:
            raise RuntimeError(NO_CANDIDATES_MESSAGE)
        return measured, best

    def pool_snapshot(self) -> tuple[Pool, list[PoolItem]]:
        """Return the pool and a copy of its items, taken under the lock. Raises RuntimeError for an empty pool."""
        with self._pool_lock:
            if self.pool is None or not self.pool.items:
                raise RuntimeError(NO_CANDIDATES_MESSAGE)
            return self.pool, list(self.pool.items)

    def kick(self, magnitude: float) -> dict:
        self.check_can_kick()
        self.wait_for_pool()
        pool, items = self.pool_snapshot()
        strength = bands.strength_for(magnitude)
        target = bands.target_for(magnitude)
        measured, best = self.measure_and_choose(items, target)
        if abs(best.rel - target) > OFF_TARGET_WARN_REL:
            # The pool is topped up band by band in the background; at kick time the nearest measured pick plays
            # now, and the panel says where it actually landed. Waiting on the brain here cost a minute per kick.
            self.log(f"kick lands off target: best rel {best.rel:.2f} vs {target:.2f} among {len(items)}")
        return self.send_on(pool, items, measured, best, strength=strength, magnitude=magnitude, target=target)

    def kick_pick(self, cand_id: int) -> dict:
        """Kick one named candidate. Its strength is the band it measures into."""
        self.check_can_kick()
        pool, items = self.pool_snapshot()
        index = next((position for position, item in enumerate(items) if item.cand_id == cand_id), None)
        if index is None:
            raise RuntimeError("that sub has left the bench; pick another")
        measured = self.measure_pool(items)
        best = measured[index]
        strength = best.band
        return self.send_on(pool, items, measured, best, strength=strength,
                            magnitude=bands.STRENGTH_MAGNITUDE[strength], target=bands.TARGET_REL[strength])

    def check_can_kick(self) -> None:
        if self.state.vector is None:
            raise RuntimeError(NO_STATE_MESSAGE)
        if self.playing_now_or_none() is None:
            # A Spotify that reports 'stopped' (just launched, or idle) takes `play track` without ever starting.
            raise RuntimeError(NOT_PLAYING_MESSAGE)
        if not self.api.configured:
            raise RuntimeError(NOT_CONFIGURED_MESSAGE)

    def send_on(
        self,
        pool: Pool,
        items: list[PoolItem],
        measured: list[bands.Measured],
        best: bands.Measured,
        *,
        strength: str,
        magnitude: float,
        target: float,
    ) -> dict:
        """Play the chosen candidate, record the kick, and make it the active kick."""
        origin = self.state.vector
        assert origin is not None  # check_can_kick / kick guard this before choosing
        chosen = items[best.index]
        pre = origin.copy()
        played = self.player.play_and_confirm(chosen.uri)
        kick_id = self.store.add_kick(strength=strength, magnitude=float(magnitude), target_rel=target,
                                      direction=chosen.direction, why=chosen.why, track_id=chosen.track.id,
                                      distance=best.distance, rel=best.rel, band=best.band, pre_state=pre,
                                      kick_vec=chosen.embedding, popularity=played.popularity)
        self.store.update_candidate(chosen.cand_id, chosen=1, kick_id=kick_id)
        self.store.add_event("kick", chosen.track.id, "kick", kick_id=kick_id, popularity=played.popularity)
        self.active = ActiveKick(id=kick_id, pre=pre, kick_vec=chosen.embedding, track_uri=chosen.uri)
        self.follow_through(chosen, played, kick_id)
        self._drop_from_pool(pool, chosen)
        summary = (f"kick #{kick_id} {strength} (target {target:.2f}) → {chosen.track.label} · "
                   f"measured {best.distance:.3f} rel {best.rel:.2f} [{best.band}] · {chosen.direction}")
        self.log(summary)
        candidates = []
        for item, measurement in zip(items, measured):
            candidates.append({"artist": item.track.artist, "title": item.track.title, "reach": item.reach,
                               "distance": measurement.distance, "rel": measurement.rel, "band": measurement.band,
                               "chosen": item is chosen})
        return {"kick_id": kick_id, "strength": strength, "target_rel": target, "track": chosen.track,
                "direction": chosen.direction, "why": chosen.why, "distance": best.distance, "rel": best.rel,
                "band": best.band, "candidates": candidates}

    def follow_through(self, chosen: PoolItem, played: spotify.Track, kick_id: int) -> None:
        """Record the kicked track as playing, so the log has it even if no observation follows."""
        self.judge_previous_track()
        self.state.update(chosen.embedding, counts_for_scale=False)
        self.store.add_event("play", chosen.track.id, "kick", popularity=played.popularity, kick_id=kick_id)
        self._last_uri = chosen.uri
        self._last_track = played
        self._max_pos = 0.0

    def _drop_from_pool(self, pool: Pool, chosen: PoolItem) -> None:
        """Remove the played item from ``pool`` if it is still the live pool."""
        with self._pool_lock:
            if self.pool is pool:
                pool.items = [item for item in pool.items if item.cand_id != chosen.cand_id]

    # ------------------------------------------------------------------- love
    def toggle_love(self) -> tuple[Track, bool]:
        """Toggle the favourite on the current track by appending a ``love`` or ``unlove`` event.

        Returns the track and whether it is now loved.
        """
        playing = self.playing_now_or_none()
        if playing is None:
            raise RuntimeError(NOT_PLAYING_MESSAGE)
        track = self.store.track_by_uri(playing.uri)
        if track is None:
            track = self.store.upsert_track(playing.artist, playing.name, album=playing.album, spotify_uri=playing.uri,
                                            duration_s=playing.duration_s)
        loved_now = not self.store.is_loved(track.id)
        self.store.add_event("love" if loved_now else "unlove", track.id, "user")
        self.log(f"{'loved' if loved_now else 'unloved'} {track.label}")
        return track, loved_now

    def step_series(self, n: int = STEPS_SHOWN) -> list[dict]:
        """Return the cosine distance from each play to the one before it, in listening order.

        Plays without an embedding are skipped, and a track logged twice in a row counts once.
        """
        steps: list[dict] = []
        previous: np.ndarray | None = None
        previous_track_id: int | None = None
        for play in self.store.play_sequence(n):
            if play["track_id"] == previous_track_id:
                continue
            vector = self.store.embedding(play["track_id"])
            if vector is None:
                continue
            previous_track_id = play["track_id"]
            if previous is not None:
                distance = float(1.0 - previous @ vector)
                label = f"{play['artist']} — {play['title']}"
                steps.append({"label": label, "source": play["source"], "kind": play["kind"], "distance": distance})
            previous = vector
        return steps

    def is_loved(self, uri: str) -> bool:
        track = self.store.track_by_uri(uri)
        return track is not None and self.store.is_loved(track.id)

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        step, far = self.state.scale()
        return {"state": {"n": len(self.state.history), "typical_step": step, "far": far},
                "pool": {"ready": self.ready(), "building": self._pool_building(), "bands": self.pool_bands(),
                         "bench": self.bench(), "lean": self.cfg.lean},
                "kick": self._kick_snapshot()}

    def spotify_picks_since(self, kick: ActiveKick) -> list[dict]:
        """Return the recommender's picks since the kick, each with its projection along the kick.

        The projection is the one the verdict uses, applied to the track rather than the listener state.
        """
        plays = [play for play in self.store.plays_since_kick(kick.id) if play["kind"] == "play"]
        embeddings = self.store.embeddings([play["track_id"] for play in plays])
        picks: list[dict] = []
        previous_track_id = None
        for play in plays:
            vector = embeddings.get(play["track_id"])
            if vector is None or play["track_id"] == previous_track_id:
                continue  # only plays the judge counted: measurable, and not the same song logged twice in a row
            previous_track_id = play["track_id"]
            along = bands.followed(kick.pre, kick.kick_vec, vector)
            picks.append({"artist": play["artist"], "title": play["title"], "along": along})
            if len(picks) == bands.SONGS_TO_JUDGE:
                break
        return picks

    def _kick_snapshot(self) -> dict | None:
        kick = self.active
        if kick is None:
            return None
        stored = self.store.kick(kick.id)
        if not stored:
            return None
        chosen = self.spotify_picks_since(kick)
        return {"id": kick.id, "strength": stored["strength"], "direction": stored["direction"] or "",
                "track": self.store.track(stored["track_id"]), "distance": stored["distance"], "rel": stored["rel"],
                "band": stored["band"], "popularity": stored["popularity"], "n_since": kick.n_since,
                "followed": kick.followed, "verdict": bands.verdict(kick.followed, kick.n_since),
                "target_rel": stored["target_rel"], "magnitude": stored["magnitude"], "chosen": chosen}
