"""The whole kick loop with fakes: a fake player, a fake brain, a fake ruler, no network."""
import threading
import time

import numpy as np
import pytest

from spotkick.brain.llm import BrainError
from spotkick.config import Config
from spotkick.ears import clap, previews
from spotkick.kick import bands, session
from spotkick.memory.store import Store
from spotkick.player.spotify import PlayerError, Track
from spotkick.player.spotify_api import SpotifyAPI

DIM = 8
HOME_SONGS = 100                # songs numbered below this cluster near vec(0) ("home"); the rest are far
URI = "spotify:track:" + "{:022d}"


def vec(seed: int) -> np.ndarray:
    return bands.normalize(np.random.default_rng(seed).standard_normal(DIM).astype(np.float32))


def song_vec(number: int) -> np.ndarray:
    """Every fake song has a deterministic vector."""
    if number < HOME_SONGS:
        return bands.normalize(vec(0) + 0.15 * vec(number + 1))
    return vec(number)


def song_number(uri: str) -> int:
    return int(uri.split(":")[-1])


def title_number(title: str) -> int:
    return int(title.split()[-1])


class FakePlayer:
    PlayerError = PlayerError

    def __init__(self):
        self.current: Track | None = None
        self.played: list[str] = []

    def set(self, number: int, position: float = 5.0, duration: float = 200.0) -> None:
        self.current = Track(name=f"Song {number}", artist=f"Artist {number}", album="", duration_s=duration,
                             position_s=position, uri=URI.format(number), popularity=40 + number % 50)

    def now_playing(self) -> Track | None:
        return self.current

    def play(self, uri: str) -> None:
        self.played.append(uri)
        self.set(song_number(uri))

    def play_and_confirm(self, uri: str, *, timeout_s: float = 8.0) -> Track:
        self.play(uri)
        if self.current is None:
            raise PlayerError(f"nothing playing after {uri}")
        return self.current


class FakeEmbedder(clap.Embedder):
    """The ruler, without the ONNX model: every fake preview URL maps to its song's vector."""

    def embed_url(self, preview_url: str) -> np.ndarray:
        return song_vec(int(preview_url.split("/")[-1]))


class FakeBrain:
    name = "fake"

    def __init__(self):
        self.calls: list[str] = []
        self.next_ids = iter(range(100, 1000))

    def complete_json(self, prompt: str, schema: dict, *, timeout: int = 240) -> dict:
        self.calls.append(prompt)
        count = 4 if "ONE direction" in prompt else 6
        candidates = []
        for _ in range(count):
            number = next(self.next_ids)
            candidates.append({"reach": ["near", "adjacent", "far"][number % 3], "direction": f"dir {number}",
                               "artist": f"Artist {number}", "title": f"Song {number}", "why": "w"})
        return {"candidates": candidates}


def fake_preview(artist: str, title: str, *, country: str = "us", session=None) -> previews.Preview:
    number = title_number(title)
    return previews.Preview(artist, title, None, f"http://p/{number}", 200.0)


def candidate_set_ids(store: Store) -> list[str]:
    """Every set the brain was asked for, oldest first."""
    rows = store._all("SELECT DISTINCT set_id FROM candidates ORDER BY id")
    return [row["set_id"] for row in rows]


class FakeSpotifyAPI(SpotifyAPI):
    """Spotify's search, faked: every "Song N" by "Artist N" exists and has the matching URI."""

    def __init__(self):
        super().__init__("id", "secret")

    def search_tracks(self, artist: str, title: str):
        from spotkick.player.spotify_api import FoundTrack

        number = title_number(title)
        return [FoundTrack(URI.format(number), f"Artist {number}", f"Song {number}")]


def new_session(config: Config, store: Store, brain: FakeBrain, player: FakePlayer) -> session.KickSession:
    return session.KickSession(config, store, brain, FakeEmbedder(), player=player, api=FakeSpotifyAPI(),
                               log=lambda message: None)


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(previews, "lookup", fake_preview)
    player = FakePlayer()
    store = Store(":memory:")
    brain = FakeBrain()
    listener = new_session(Config(alpha=0.7, n_candidates=6), store, brain, player)
    return listener, store, player, brain


def play_through(player: FakePlayer, listener: session.KickSession, number: int) -> None:
    """Let the current song finish, then start song `number` (a real listener finishes songs; the fake must too)."""
    if player.current is not None:
        current = song_number(player.current.uri)
        player.set(current, position=player.current.duration_s - 1)
        listener.observe()
    player.set(number)
    listener.observe()


def wait(condition, timeout: float = 5.0) -> bool:
    started = time.time()
    while time.time() - started < timeout:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_kick_picks_by_measured_distance_and_judges_the_continuation(world):
    listener, store, player, brain = world
    for number in range(4):                                # the listener plays four home songs
        play_through(player, listener, number)
    assert store.counts()["events"] == 4
    assert len(listener.state.history) == 4
    assert wait(lambda: listener.ready() > 0)              # prefetch happened in the background
    assert wait(lambda: not listener._pool_building())     # a band top-up may follow; let it land
    assert "Last plays" in brain.calls[0]
    first_set_id = candidate_set_ids(store)[0]
    set_rows = store.candidate_set(first_set_id)
    assert len(set_rows) == 6
    assert all(row["track_id"] for row in set_rows)

    out = listener.kick(0.9)                               # boot
    assert out["strength"] == "boot"
    assert out["track"].spotify_uri == player.played[0]
    by_target = sorted(out["candidates"], key=lambda candidate: abs(candidate["rel"] - out["target_rel"]))
    assert by_target[0]["chosen"]                          # the one nearest the target on the listener's own scale
    kick = store.last_kick()
    assert kick["strength"] == "boot"
    assert kick["band"] == out["band"]
    assert kick["popularity"] == player.current.popularity
    assert store.candidate_set(set_rows[0]["set_id"])[0]["distance"] is not None

    events = store.events(kinds=("kick", "play"))
    assert events[-1]["source"] == "kick"
    assert events[-1]["kick_id"] == kick["id"]
    assert events[-1]["kind"] == "play"
    plays_before = [event for event in events if event["kind"] == "play"]
    listener.observe()                                     # the kick track is already counted; observing adds nothing
    assert len(store.events(kinds=("play",))) == len(plays_before)
    assert listener.active.n_since == 0

    for number in (5, 6):                                  # Spotify drifts back home
        play_through(player, listener, number)
    assert listener.active.n_since == 2
    snap = listener.snapshot()["kick"]
    assert snap["verdict"] in ("returned", "bent")
    assert [pick["title"] for pick in snap["chosen"]] == ["Song 5", "Song 6"]
    assert all(pick["along"] is not None and pick["along"] < 0.6 for pick in snap["chosen"])   # home songs: near 0
    assert snap["n_since"] == 2
    assert store.kick(kick["id"])["verdict"] == snap["verdict"]
    frozen = (snap["followed"], snap["verdict"])
    play_through(player, listener, 7)                      # a third song plays: the verdict is already in
    assert listener.active.n_since == 2
    later = listener.snapshot()["kick"]
    assert (later["followed"], later["verdict"]) == frozen
    # the pool lost the chosen item and gets rebuilt for the new context
    assert wait(lambda: listener.ready() > 0)


def test_skip_is_recorded_when_a_song_is_abandoned_early(world):
    listener, store, player, _ = world
    player.set(1, position=3.0)
    listener.observe()
    player.set(1, position=20.0)
    listener.observe()
    player.set(2, position=1.0)
    listener.observe()
    kinds = [event["kind"] for event in store.events()]
    assert kinds == ["play", "skip", "play"]
    assert store.rejected() == ["Artist 1 — Song 1"]


def test_kick_needs_a_state_and_waits_for_the_pool(world):
    listener, _, player, _ = world
    with pytest.raises(RuntimeError):
        listener.kick(0.5)
    player.set(1)
    listener.observe()
    out = listener.kick(0.1)                               # pool may still be building → waits for it
    assert out["strength"] == "tap"
    assert listener.active.n_since == 0


def test_state_and_pool_are_rebuilt_from_the_store(world):
    listener, store, player, brain = world
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0)
    assert wait(lambda: not listener._pool_building())        # a band top-up may be in flight; let it land
    brain_calls = len(brain.calls)
    again = new_session(Config(), store, brain, player)
    assert len(again.state.history) == 3
    assert again.state.vector is not None and np.allclose(again.state.vector, listener.state.vector)
    assert again.ready() == listener.ready()
    assert len(brain.calls) == brain_calls                 # restored, not re-asked
    again.kick(0.5)
    assert again.active is not None
    third = new_session(Config(), store, brain, player)
    assert third.active is not None
    assert third.active.id == again.active.id
    assert third.active.track_uri == player.played[-1]
    events_before = store.counts()["events"]
    third.observe()                                        # the kick track is playing and already logged: nothing new
    assert store.counts()["events"] == events_before
    player.set(7)                                          # Spotify moved on while nobody was watching...
    fourth = new_session(Config(), store, brain, player)
    fourth.observe()                                       # ...so the song playing at startup is counted, and judged
    assert store.counts()["events"] == events_before + 1
    assert fourth.active is not None and fourth.active.n_since == 1


def test_pool_build_is_threadsafe_with_observe(world):
    listener, store, player, _ = world
    player.set(1)
    listener.observe()
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            listener.observe()
            time.sleep(0.005)

    poller = threading.Thread(target=poll)
    poller.start()
    try:
        assert wait(lambda: listener.ready() > 0)
        listener.kick(0.5)
    finally:
        stop.set()
        poller.join()
    assert store.counts()["kicks"] == 1


def test_kick_never_waits_on_the_brain_when_a_pool_exists(world, monkeypatch):
    """The pool is topped up in the background; a kick plays the nearest measured pick now, even off target."""
    listener, _store, player, brain = world
    for number in range(4):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0)
    assert wait(lambda: not listener._pool_building())
    log_lines: list[str] = []
    listener.log = log_lines.append
    calls_before = len(brain.calls)
    started = time.time()
    out = listener.kick(0.1)                                   # a tap against a pool that only measures far
    assert time.time() - started < 1.0
    assert out["track"].spotify_uri == player.played[0]
    assert len(brain.calls) == calls_before
    assert any("lands off target" in line for line in log_lines)


def test_prefetch_survives_a_brain_that_is_rate_limited(world, monkeypatch):
    """A background prefetch has nobody to catch its exceptions; a capped brain must not kill the thread."""
    listener, _store, player, brain = world
    log_lines: list[str] = []
    listener.log = log_lines.append

    def brain_is_capped(prompt: str, schema: dict, *, timeout: int = 240) -> dict:
        raise BrainError("codex failed: You've hit your usage limit.")

    monkeypatch.setattr(brain, "complete_json", brain_is_capped)
    for number in range(4):
        play_through(player, listener, number)
    assert wait(lambda: any("prefetch skipped" in line for line in log_lines))
    assert listener.ready() == 0


def test_an_empty_band_is_topped_up_in_the_background(world):
    """The fake brain's picks all measure far, so after the first set the tap and kick bands are empty: the next
    observation asks for a 'near' top-up (one band at a time), merges it, then moves to the next empty band before
    coming back to correct the first."""
    listener, store, player, brain = world
    for number in range(4):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0)
    first_ready = listener.ready()
    coverage = listener.pool_bands()
    assert coverage["boot"] == first_ready and coverage["tap"] == 0

    listener.observe()                                         # nothing new playing; the pool has an empty band
    assert wait(lambda: len(brain.calls) >= 2)
    assert "all labelled 'near'" in brain.calls[1]
    assert wait(lambda: listener.ready() > first_ready)        # merged, not replaced
    assert len(candidate_set_ids(store)) == 2

    listener.observe()                                         # tap is still empty (the picks measured far)...
    assert wait(lambda: len(brain.calls) >= 3)
    assert "all labelled 'adjacent'" in brain.calls[2]         # ...so the next band is asked for, not tap again
    assert wait(lambda: not listener._pool_building())
    listener.observe()                                         # both bands have missed once: tap gets a correction
    assert wait(lambda: len(brain.calls) >= 4)
    assert "all labelled 'near'" in brain.calls[3]
    assert "did not land" in brain.calls[3]                    # ...told where its earlier picks measured
    assert wait(lambda: not listener._pool_building())

    again = new_session(Config(), store, brain, player)        # a restart restores the merged pool
    assert again.ready() == listener.ready()


def test_a_track_proposed_twice_is_in_the_pool_once(world):
    listener, store, player, brain = world
    for number in range(4):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    first = store.library_candidates()[-1]
    duplicate_rows = [{"reach": "far", "direction": "again", "artist": first["artist"], "title": first["title"],
                       "why": "", "track_id": first["track_id"]}]
    store.add_candidates("dup-set", duplicate_rows)                # the same song, proposed in a second set
    again = new_session(Config(), store, brain, player)
    assert again.pool is not None
    track_ids = [item.track.id for item in again.pool.items]
    assert len(track_ids) == len(set(track_ids))
    assert again.ready() == listener.ready()


def test_kick_refuses_when_spotify_is_not_playing_here(world):
    """A Spotify that reports 'stopped' (just launched, or idle) accepts `play track` without ever starting, so the
    kick must say so up front instead of timing out on the confirmation."""
    listener, _store, player, _brain = world
    for number in range(2):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0)
    player.current = None
    with pytest.raises(RuntimeError, match="nothing playing on this Mac"):
        listener.kick(0.5)
    assert player.played == []


def test_without_credentials_the_brain_is_never_asked(monkeypatch):
    monkeypatch.setattr(previews, "lookup", fake_preview)
    player = FakePlayer()
    store = Store(":memory:")
    brain = FakeBrain()
    log_lines: list[str] = []
    listener = session.KickSession(Config(), store, brain, FakeEmbedder(), player=player,
                                   api=SpotifyAPI("", ""), log=log_lines.append)
    for number in range(3):
        play_through(player, listener, number)
    time.sleep(0.1)
    assert brain.calls == []
    assert any("no Spotify credentials" in line for line in log_lines)
    with pytest.raises(RuntimeError, match="no Spotify credentials"):
        listener.kick(0.5)


def test_a_restart_does_not_log_the_playing_song_again(world):
    """Each relaunch used to re-ingest whatever was playing: three restarts, three plays of one song, step zero."""
    listener, store, player, brain = world
    play_through(player, listener, 1)
    assert len(store.events(kinds=("play",))) == 1
    again = listener
    for _ in range(3):
        again = new_session(Config(), store, brain, player)
        again.observe()
    assert len(store.events(kinds=("play",))) == 1
    assert len(again.state.history) == 1
    player.set(2)                                              # a genuinely new song after a restart is counted
    again.observe()
    assert len(store.events(kinds=("play",))) == 2


def test_love_toggles_and_the_latest_event_wins(world):
    listener, store, player, _brain = world
    with pytest.raises(RuntimeError):
        listener.toggle_love()                                 # nothing playing
    player.set(1)
    listener.observe()
    track, loved = listener.toggle_love()
    assert (track.artist, loved) == ("Artist 1", True)
    assert store.loved() == ["Artist 1 — Song 1"]
    assert listener.is_loved(player.current.uri)
    _track, loved = listener.toggle_love()                     # pressing again takes it back
    assert not loved
    assert store.loved() == []
    assert not listener.is_loved(player.current.uri)
    _track, loved = listener.toggle_love()                     # and again: loved once more
    assert loved
    assert store.loved() == ["Artist 1 — Song 1"]
    assert [event["kind"] for event in store.events(kinds=("love", "unlove"))] == ["love", "unlove", "love"]
    assert not listener.is_loved(URI.format(2))


def test_a_new_lean_drops_the_pool_and_reaches_the_brain(world):
    listener, _store, player, brain = world
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    listener.cfg.lean = "melancholic, Portuguese"
    listener.invalidate_pool()
    assert listener.pool is None
    listener.observe()
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    assert 'stay inside it: "melancholic, Portuguese"' in brain.calls[-1]


def test_a_build_in_flight_when_the_lean_changes_is_thrown_away(world):
    listener, _store, player, brain = world
    gate = threading.Event()
    original = brain.complete_json

    def slow_complete_json(prompt: str, schema: dict, *, timeout: int = 240) -> dict:
        gate.wait(5.0)
        return original(prompt, schema, timeout=timeout)

    brain.complete_json = slow_complete_json                   # type: ignore[method-assign]
    for number in range(3):
        play_through(player, listener, number)                 # starts a build that now hangs in the brain
    assert wait(lambda: listener._pool_building())
    listener.cfg.lean = "jazz, calm"
    listener.invalidate_pool()
    gate.set()                                                 # the old-prompt build finishes...
    assert wait(lambda: not listener._pool_building())
    assert listener.pool is None                               # ...and installs nothing
    listener.observe()
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    assert 'stay inside it: "jazz, calm"' in brain.calls[-1]


def test_the_pool_refills_from_the_library_before_asking_the_brain(world):
    """Every candidate ever resolved is in the store with its embedding; a dropped pool comes back from there, and
    only the bands the library leaves empty go to the brain."""
    listener, store, player, brain = world
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    known = {item.track.id for item in listener.pool.items}
    calls_before = len(brain.calls)
    listener.invalidate_pool()                                 # same lean: nothing new to ask for
    listener.observe()
    assert listener.pool is not None and listener.pool.set_id == session.LIBRARY_SET_ID
    assert {item.track.id for item in listener.pool.items} == known
    assert wait(lambda: not listener._pool_building())
    top_ups = brain.calls[calls_before:]                       # the brain is only asked for a band the library lacks
    assert all("all labelled '" in prompt for prompt in top_ups)
    assert len(store.library_candidates("")) >= len(known)

    listener.cfg.lean = "jazz, calm"                           # another lean: the library has nothing for it
    listener.invalidate_pool()
    listener.observe()
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    assert len(brain.calls) > calls_before
    assert listener.pool.set_id != session.LIBRARY_SET_ID
    assert all(row["lean"] == "jazz, calm" for row in store.library_candidates("jazz, calm"))


def test_a_kicked_song_leaves_the_library(world):
    listener, store, player, _brain = world
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    before = {row["track_id"] for row in store.library_candidates("")}
    out = listener.kick(0.5)
    after = {row["track_id"] for row in store.library_candidates("")}
    assert before - after == {out["track"].id}
    again = new_session(Config(), store, _brain, player)      # a restart restores the library, minus the kick
    assert again.pool is not None and again.pool.set_id == session.LIBRARY_SET_ID
    assert {item.track.id for item in again.pool.items} == after


def test_the_bench_shows_every_pick_by_band_nearest_target_first(world):
    listener, _store, player, _brain = world
    assert all(band["state"] == "empty" and band["picks"] == [] for band in listener.bench().values())
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    bench = listener.bench()
    assert sum(len(band["picks"]) for band in bench.values()) == listener.ready()
    for name, band in bench.items():
        assert band["state"] in ("ready", "resting", "empty")
        assert (band["state"] == "ready") == bool(band["picks"])
        distances = [pick["distance"] for pick in band["picks"]]
        assert distances == sorted(distances)                   # nearest first, the ruler's order
        if band["picks"]:
            target = bands.TARGET_REL[name]
            gaps = [abs(pick["rel"] - target) for pick in band["picks"]]
            assert abs(band["would_play"]["rel"] - target) == min(gaps)
    assert listener.snapshot()["pool"]["bench"].keys() == {"tap", "kick", "boot"}


def test_an_empty_band_is_retried_with_the_misses_fed_back(world):
    """The brain keeps naming songs that measure close; the harness keeps asking, telling it where each landed."""
    listener, _store, player, brain = world
    brain.next_ids = iter(range(10, 99))                      # every proposal is a home song: a tap, never a boot
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    assert listener.pool_bands()["boot"] == 0
    for _ in range(5):                                         # kick and boot each get a first ask, then corrections
        listener.observe()
        assert wait(lambda: not listener._pool_building())
    assert listener._band_attempts["boot"] >= 2
    corrections = [prompt for prompt in brain.calls if "did not land" in prompt]
    assert corrections, "the retry must tell the brain where its picks measured"
    assert "measured as a small step" in corrections[-1]
    assert listener.bench()["boot"]["state"] == "resting"       # third try waits its turn, not ten minutes
    listener.cfg.lean = "country"                              # the same misses under a lean: the lean is the cap
    assert listener.bench()["boot"]["state"] == "capped"
    listener.cfg.lean = ""
    assert listener.band_retry_delay("boot") == session.BAND_RETRY_DELAYS_S[2]
    listener.invalidate_pool()                                  # a settings change starts the count over
    assert listener._band_attempts == {} and listener._band_misses == {}


def test_pointing_at_a_sub_sends_that_one_on(world):
    listener, store, player, _brain = world
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    bench = listener.bench()
    band, pick = next((band, entry["picks"][-1]) for band, entry in bench.items() if entry["picks"])
    out = listener.kick_pick(pick["cand_id"])
    assert out["track"].artist == pick["artist"] and player.played[-1] == out["track"].spotify_uri
    assert out["strength"] == band                              # the strength is what the pick measures as
    assert store.last_kick()["strength"] == band
    assert all(item.cand_id != pick["cand_id"] for item in listener.pool.items)   # off the bench, on the pitch
    with pytest.raises(RuntimeError):
        listener.kick_pick(pick["cand_id"])                     # cannot be sent on twice


def test_step_series_is_the_distance_between_consecutive_plays(world):
    listener, _store, player, _brain = world
    assert listener.step_series() == []
    for number in range(4):
        play_through(player, listener, number)
    steps = listener.step_series()
    assert len(steps) == 3                                     # four plays, three transitions, oldest first
    assert all(0.0 <= step["distance"] <= 2.0 for step in steps)
    assert all(step["source"] == "spotify" for step in steps)
    assert wait(lambda: listener.ready() > 0 and not listener._pool_building())
    listener.kick(0.5)
    assert listener.step_series()[-1]["source"] == "kick"
    steps_after_kick = len(listener.step_series())
    listener.kick(0.5)                                         # the kicked song is skipped away from: not a step
    assert listener.step_series()[-1]["source"] == "kick"
    assert len(listener.step_series()) == steps_after_kick + 1


def test_kicked_songs_move_the_state_but_not_the_ruler(world):
    """Sixteen of twenty recent plays being kicks made a 'typical step' the size of a kick, so nothing could ever
    measure far. The scale is Spotify's spread; the kicks are the intervention."""
    listener, store, player, _brain = world
    for number in range(3):
        play_through(player, listener, number)
    assert wait(lambda: listener.ready() > 0)
    assert wait(lambda: not listener._pool_building())
    before = listener.state.vector.copy()
    scale_before = listener.state.scale()
    listener.kick(0.9)                                         # a far song plays
    assert not np.allclose(listener.state.vector, before)      # the state moved
    assert len(listener.state.history) == 3                    # the ruler did not learn from it
    assert listener.state.scale() == scale_before
    again = new_session(Config(), store, _brain, player)       # and a restart rebuilds it the same way
    assert len(again.state.history) == 3
    assert again.state.scale() == scale_before
