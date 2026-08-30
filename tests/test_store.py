import sqlite3
import time

import numpy as np
import pytest

from spotkick.memory.store import Store, track_key

URI = "spotify:track:0000000000000000000000"
SECONDS_PER_DAY = 86400


@pytest.fixture
def store():
    return Store(":memory:")


def add_kick_candidate(
    store: Store,
    track,
    *,
    direction: str = "brazilian soul",
    why: str = "",
    distance: float = 0.5,
    rel: float = 1.2,
    band: str = "boot",
) -> int:
    candidate_id = store.add_candidates(
        f"kick-{track.id}",
        [{"reach": "far", "direction": direction, "artist": track.artist, "title": track.title, "why": why}],
    )[0]
    store.update_candidate(candidate_id, track_id=track.id, distance=distance, rel=rel, band=band)
    return candidate_id


def test_track_key_normalizes():
    assert track_key("Fela Kuti", "Water No Get Enemy") == track_key("fela  kuti", "Water no get enemy!")


def test_upsert_track_enriches_without_overwriting(store):
    first = store.upsert_track("Azymuth", "Linha do Horizonte")
    enriched = store.upsert_track("Azymuth", "Linha do Horizonte", spotify_uri=URI, duration_s=270)
    assert enriched.id == first.id
    assert enriched.spotify_uri == URI
    assert enriched.duration_s == 270
    again = store.upsert_track("azymuth", "linha do horizonte", duration_s=999)
    assert again.duration_s == 270  # first value wins
    assert store.track_by_uri(URI).id == first.id


def test_embedding_roundtrip(store):
    track = store.upsert_track("Stereolab", "Cybele's Reverie")
    vector = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    store.put_embedding(track.id, vector, "clap")
    assert np.allclose(store.embedding(track.id), vector)
    assert set(store.embeddings([track.id, 999])) == {track.id}
    assert store.embedding(999) is None


def test_seen_covers_plays_kicks_and_picks(store):
    played = store.upsert_track("Tinariwen", "Amassakoul")
    assert not store.seen("Tinariwen", "Amassakoul")
    store.add_event("play", played.id, "spotify")
    assert store.seen("tinariwen", "amassakoul")
    kicked = store.upsert_track("Ed Motta", "Manuel")
    candidate_id = add_kick_candidate(store, kicked)
    store.add_kick(
        candidate_id=candidate_id,
        strength="boot",
        magnitude=0.9,
        target_rel=1.3,
        pre_state=np.zeros(4),
    )
    assert store.seen("Ed Motta", "Manuel")
    assert not store.seen("Nobody", "Nothing")


def test_context_queries(store):
    now = time.time()
    zombie = store.upsert_track("Fela Kuti", "Zombie")
    water = store.upsert_track("Fela Kuti", "Water No Get Enemy")
    ping_pong = store.upsert_track("Stereolab", "Ping Pong")
    old_song = store.upsert_track("Old Band", "Old Song")
    store.add_event("play", old_song.id, "spotify", t=now - 90 * SECONDS_PER_DAY)
    store.add_event("play", old_song.id, "spotify", t=now - 89 * SECONDS_PER_DAY)
    store.add_event("play", old_song.id, "spotify", t=now - 88 * SECONDS_PER_DAY)
    store.add_event("play", zombie.id, "spotify", t=now - 3000)
    store.add_event("play", water.id, "kick", t=now - 2000)
    store.add_event("partial", ping_pong.id, "spotify", t=now - 1000, completion=0.4)
    store.add_event("skip", ping_pong.id, "spotify", t=now - 500, skip_at_s=20)
    store.add_event("love", zombie.id, "user", t=now - 100)

    recent = store.recent(2)
    assert [play["title"] for play in recent] == ["Ping Pong", "Ping Pong"]
    assert recent[0]["kind"] == "skip"
    assert store.top_artists(days=30, n=5)[0] == ("Fela Kuti", 2)
    assert store.top_artists(n=1)[0] == ("Old Band", 3)
    assert store.loved() == ["Fela Kuti — Zombie"]
    assert store.rejected() == ["Stereolab — Ping Pong"]

    kicked = store.upsert_track("Ed Motta", "Manuel")
    candidate_id = add_kick_candidate(store, kicked, distance=0.4, rel=0.7, band="kick")
    kick_id = store.add_kick(
        candidate_id=candidate_id,
        strength="kick",
        magnitude=0.5,
        target_rel=0.75,
        pre_state=np.zeros(4),
        t=now - 50,
    )
    store.add_event("kick", kicked.id, "kick", kick_id=kick_id, t=now - 50)
    store.add_event("play", kicked.id, "kick", t=now - 49)  # the kicked track: not counted
    followed_by = store.upsert_track("Jorge Ben Jor", "Taj Mahal")
    store.add_event("play", followed_by.id, "spotify", t=now - 10)
    assert store.directions() == ["brazilian soul"]
    assert store.kicked_artists() == ["Ed Motta"]
    plays_after = store.plays_since_kick(kick_id)
    assert [play["artist"] for play in plays_after] == ["Jorge Ben Jor"]
    store.update_kick(kick_id, followed=0.7, verdict="followed", n_since=1)
    assert store.last_kick()["verdict"] == "followed"
    with pytest.raises(ValueError):
        store.update_kick(kick_id, distance=0.1)


def test_candidates_lifecycle(store):
    proposed = [
        {"reach": "near", "direction": "d1", "artist": "A", "title": "1", "why": "", "spotify_uri": "spotify:track:x"},
        {"reach": "far", "direction": "d2", "artist": "B", "title": "2", "why": ""},
        {"reach": "far", "direction": "d3", "artist": "C", "title": "3", "why": "", "rejected_reason": "seen"},
    ]
    candidate_ids = store.add_candidates("set-1", proposed)
    track = store.upsert_track("A", "1")
    store.update_candidate(candidate_ids[0], track_id=track.id, distance=0.2, rel=0.3, band="tap")
    usable = store.library_candidates()
    assert [candidate["artist"] for candidate in usable] == ["A"]  # B unresolved, C rejected
    store.update_candidate(candidate_ids[0], chosen=1)
    assert store.library_candidates() == []
    assert len(store.candidate_set("set-1")) == 3
    assert store.counts()["candidates"] == 3


def test_kick_schema_links_request_to_candidate_and_events(store):
    assert store.columns("kicks") == {
        "id", "t", "candidate_id", "selected_by", "nearest_candidate_id", "minime_score", "strength",
        "magnitude", "target_rel", "pre_state", "followed", "verdict", "n_since",
    }
    assert "kick_id" not in store.columns("candidates")
    kicked = store.upsert_track("Ed Motta", "Manuel")
    kick_vector = np.ones(4, dtype=np.float32)
    store.put_embedding(kicked.id, kick_vector, "clap")
    candidate_id = add_kick_candidate(store, kicked, distance=0.4, rel=0.7, band="kick")
    kick_id = store.add_kick(candidate_id=candidate_id, selected_by="mini-me",
                             nearest_candidate_id=candidate_id, minime_score=0.73, strength="kick", magnitude=0.5,
                             target_rel=0.75, pre_state=np.zeros(4))
    store.update_candidate(candidate_id, chosen=1)
    store.add_event("kick", kicked.id, "kick", kick_id=kick_id, popularity=42)

    kick = store.kick(kick_id)
    assert kick["candidate_id"] == candidate_id
    assert kick["selected_by"] == "mini-me"
    assert kick["nearest_candidate_id"] == candidate_id
    assert kick["minime_score"] == 0.73
    assert kick["track_id"] == kicked.id
    assert kick["direction"] == "brazilian soul"
    assert kick["distance"] == 0.4
    assert kick["popularity"] == 42
    assert np.array_equal(kick["kick_vec"], kick_vector)


def test_persists_to_disk(tmp_path):
    path = tmp_path / "x" / "spotkick.db"
    writer = Store(path)
    track = writer.upsert_track("A", "1")
    writer.add_event("play", track.id, "spotify")
    writer.close()
    reader = Store(path)
    assert reader.counts() == {"tracks": 1, "embeddings": 0, "events": 1, "kicks": 0, "candidates": 0}


def test_adds_selection_audit_columns_to_the_previous_v0_schema(tmp_path):
    path = tmp_path / "previous.db"
    writer = Store(path)
    kicked = writer.upsert_track("Ed Motta", "Manuel")
    candidate_id = add_kick_candidate(writer, kicked)
    kick_id = writer.add_kick(candidate_id=candidate_id, strength="kick", magnitude=0.5, target_rel=0.75,
                              pre_state=np.zeros(4))
    writer.close()

    with sqlite3.connect(path) as database:
        database.execute("ALTER TABLE kicks DROP COLUMN minime_score")
        database.execute("ALTER TABLE kicks DROP COLUMN nearest_candidate_id")
        database.execute("ALTER TABLE kicks DROP COLUMN selected_by")

    upgraded = Store(path)
    assert {"selected_by", "nearest_candidate_id", "minime_score"} <= upgraded.columns("kicks")
    legacy = upgraded.kick(kick_id)
    assert legacy is not None
    assert legacy["selected_by"] == "legacy"
    assert legacy["nearest_candidate_id"] is None and legacy["minime_score"] is None
    assert upgraded.recent_kicks()[0]["selection_changed"] is None


def test_recent_kicks_and_spotify_play_count(store):
    kicked = store.upsert_track("Ed Motta", "Manuel")
    other = store.upsert_track("Azymuth", "Partido Alto")
    candidate_id = add_kick_candidate(store, kicked)
    kick_id = store.add_kick(candidate_id=candidate_id, strength="boot", magnitude=0.9, target_rel=1.3,
                             pre_state=np.zeros(4))
    store.add_event("play", kicked.id, "kick", kick_id=kick_id)          # the kicked track
    store.add_event("play", other.id, "spotify")
    store.add_event("play", other.id, "spotify")
    store.update_kick(kick_id, followed=0.7, verdict="followed", n_since=2)
    assert store.spotify_play_count() == 2
    recent = store.recent_kicks()
    assert [kick["artist"] for kick in recent] == ["Ed Motta"]
    assert recent[0]["target_rel"] == 1.3
    assert recent[0]["selected_by"] == "nearest"
    assert recent[0]["selection_changed"] == 0
    assert recent[0]["verdict"] == "followed"
    assert recent[0]["n_since"] == 2
