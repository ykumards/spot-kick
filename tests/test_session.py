"""The whole kick loop with fakes: a fake player, a fake brain, a fake ruler, no network."""
import threading
import time

import numpy as np
import pytest

from spotkick.config import Config
from spotkick.ears import previews
from spotkick.kick import bands as B
from spotkick.kick import session as S
from spotkick.mind.store import Store
from spotkick.player.spotify import PlayerError, Track

DIM = 8


def vec(seed):
    return B.normalize(np.random.default_rng(seed).standard_normal(DIM).astype(np.float32))


# Every fake song has a deterministic vector: songs 0..9 cluster near vec(0) ("home"), 100+ are far.
def song_vec(n: int):
    if n < 100:
        return B.normalize(vec(0) + 0.15 * vec(n + 1))
    return vec(n)


URI = "spotify:track:" + "{:022d}"


class FakePlayer:
    PlayerError = PlayerError

    def __init__(self):
        self.current: Track | None = None
        self.played: list[str] = []

    def set(self, n: int, position=5.0, duration=200.0):
        self.current = Track(name=f"Song {n}", artist=f"Artist {n}", album="", duration_s=duration, position_s=position,
                             uri=URI.format(n), popularity=40 + n % 50)

    def now_playing(self):
        return self.current

    def play(self, uri):
        self.played.append(uri)
        n = int(uri.split(":")[-1])
        self.set(n)

    def play_and_confirm(self, uri, timeout_s=8.0):
        self.play(uri)
        return self.current


class FakeEmbedder:
    loaded = True

    def embed_url(self, url):
        return song_vec(int(url.split("/")[-1]))


class FakeBrain:
    name = "fake"

    def __init__(self):
        self.calls = []
        self.next_ids = iter(range(100, 1000))

    def complete_json(self, prompt, schema, *, timeout=240):
        self.calls.append(prompt)
        n = 4 if "ONE direction" in prompt else 6
        ids = [next(self.next_ids) for _ in range(n)]
        return {"candidates": [{"reach": ["near", "adjacent", "far"][i % 3], "direction": f"dir {i}", "artist": f"Artist {i}",
                                "title": f"Song {i}", "why": "w", "spotify_uri": ""} for i in ids]}

    def search_uri(self, artist, title):
        return URI.format(int(title.split()[-1]))


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(previews, "lookup", lambda artist, title, session=None, country="us":
                        previews.Preview(artist, title, None, int(title.split()[-1]), f"http://p/{title.split()[-1]}", 200.0, None))
    monkeypatch.setattr(S.RS, "oembed_title", lambda uri, session=None: f"Song {int(uri.split(':')[-1])}")
    player = FakePlayer()
    store = Store(":memory:")
    brain = FakeBrain()
    sess = S.KickSession(Config(alpha=0.7, n_candidates=6), store, brain, FakeEmbedder(), player=player, log=lambda m: None)
    return sess, store, player, brain


def play_through(player, sess, n):
    """Let the current song finish, then start song n (a real listener finishes songs; the fake must too)."""
    if player.current is not None:
        cur = int(player.current.uri.split(":")[-1])
        player.set(cur, position=player.current.duration_s - 1); sess.observe()
    player.set(n); sess.observe()


def wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_kick_picks_by_measured_distance_and_judges_the_continuation(world):
    sess, store, player, brain = world
    for n in range(4):                                     # the listener plays four home songs
        play_through(player, sess, n)
    assert store.counts()["events"] == 4 and len(sess.state.history) == 4
    assert wait(lambda: sess.ready() > 0)                  # prefetch happened in the background
    assert len(brain.calls) == 1 and "Last plays" in brain.calls[0]
    set_rows = store.latest_candidate_set(usable_only=False)
    assert len(set_rows) == 6 and all(r["track_id"] for r in set_rows)

    out = sess.kick(0.9)                                   # boot
    assert out["strength"] == "boot" and out["dose"] == 5 and out["track"].spotify_uri == player.played[0]
    ms = sorted(out["candidates"], key=lambda c: abs(c["rel"] - out["target_rel"]))
    assert ms[0]["chosen"]                                 # the one nearest the target on the listener's own scale
    k = store.last_kick()
    assert k["strength"] == "boot" and k["band"] == out["band"] and k["popularity"] == player.current.popularity
    assert store.candidate_set(k and set_rows[0]["set_id"])[0]["distance"] is not None

    ev = store.events(kinds=("kick", "play"))
    assert ev[-1]["source"] == "kick" and ev[-1]["kick_id"] == k["id"] and ev[-1]["kind"] == "play"
    sess.observe()                                         # the kick track is already counted; observing again adds nothing
    assert len(store.events(kinds=("play",))) == len([e for e in ev if e["kind"] == "play"]) and sess.active.n_since == 0
    assert wait(lambda: len(sess.active.forced_uris) == 4)  # follow-through fetched with the direction hint
    assert any("ONE direction" in c for c in brain.calls)

    player.set(int(player.played[0].split(":")[-1]), position=197.0)   # kick song ending → next forced
    sess.observe()
    assert len(player.played) == 2 and len(sess.active.forced_uris) == 3
    sess.observe()                                         # forced track started: counted as source=kick, not "since"
    assert store.events()[-1]["source"] == "kick" and sess.active.n_since == 0

    for n in (5, 6):                                       # Spotify drifts back home
        play_through(player, sess, n)
    assert sess.active.n_since == 2
    snap = sess.snapshot()["kick"]
    assert snap["verdict"] in ("returned", "bent") and snap["n_since"] == 2 and store.kick(k["id"])["verdict"] == snap["verdict"]
    # the pool lost the chosen item and gets rebuilt for the new context
    assert wait(lambda: sess.ready() > 0)


def test_skip_is_recorded_when_a_song_is_abandoned_early(world):
    sess, store, player, _ = world
    player.set(1, position=3.0); sess.observe()
    player.set(1, position=20.0); sess.observe()
    player.set(2, position=1.0); sess.observe()
    kinds = [e["kind"] for e in store.events()]
    assert kinds == ["play", "skip", "play"] and store.rejected() == ["Artist 1 — Song 1"]


def test_kick_needs_a_state_and_waits_for_the_pool(world):
    sess, _, player, _ = world
    with pytest.raises(RuntimeError):
        sess.kick(0.5)
    player.set(1); sess.observe()
    out = sess.kick(0.1)                                   # pool may still be building → waits for it
    assert out["strength"] == "tap" and out["dose"] == 1 and sess.active.forced_uris == []
    assert wait(lambda: not (sess._follow_thread and sess._follow_thread.is_alive()))


def test_state_and_pool_are_rebuilt_from_the_store(world):
    sess, store, player, brain = world
    for n in range(3):
        play_through(player, sess, n)
    assert wait(lambda: sess.ready() > 0)
    n_calls = len(brain.calls)
    again = S.KickSession(Config(), store, brain, FakeEmbedder(), player=player, log=lambda m: None)
    assert len(again.state.history) == 3 and np.allclose(again.state.vector, sess.state.vector)
    assert again.ready() == sess.ready() and len(brain.calls) == n_calls   # restored, not re-asked
    again.kick(0.5)
    assert again.active is not None
    third = S.KickSession(Config(), store, brain, FakeEmbedder(), player=player, log=lambda m: None)
    assert third.active is not None and third.active.id == again.active.id and third.active.track_uri == player.played[-1]
    n_events = store.counts()["events"]
    third.observe()                                        # the kick track is playing and already logged: nothing new
    assert store.counts()["events"] == n_events
    player.set(7)                                          # Spotify moved on while nobody was watching...
    fourth = S.KickSession(Config(), store, brain, FakeEmbedder(), player=player, log=lambda m: None)
    fourth.observe()                                       # ...so the song playing at startup is counted, and judged
    assert store.counts()["events"] == n_events + 1 and fourth.active.n_since == 1


def test_pool_build_is_threadsafe_with_observe(world):
    sess, store, player, _ = world
    player.set(1); sess.observe()
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            sess.observe(); time.sleep(0.005)
    th = threading.Thread(target=poll); th.start()
    try:
        assert wait(lambda: sess.ready() > 0)
        sess.kick(0.5)
    finally:
        stop.set(); th.join()
    assert store.counts()["kicks"] == 1
