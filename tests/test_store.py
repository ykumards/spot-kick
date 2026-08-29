import sqlite3
import time

import numpy as np
import pytest

from spotkick.mind.store import Store, track_key


@pytest.fixture
def s():
    return Store(":memory:")


def test_track_key_normalizes():
    assert track_key("Fela Kuti", "Water No Get Enemy") == track_key("fela  kuti", "Water no get enemy!")


def test_upsert_track_enriches_without_overwriting(s):
    a = s.upsert_track("Azymuth", "Linha do Horizonte")
    b = s.upsert_track("Azymuth", "Linha do Horizonte", spotify_uri="spotify:track:0000000000000000000000", duration_s=270)
    assert a.id == b.id and b.spotify_uri == "spotify:track:0000000000000000000000" and b.duration_s == 270
    c = s.upsert_track("azymuth", "linha do horizonte", duration_s=999)
    assert c.duration_s == 270  # first value wins
    assert s.track_by_uri("spotify:track:0000000000000000000000").id == a.id


def test_embedding_roundtrip(s):
    t = s.upsert_track("Stereolab", "Cybele's Reverie")
    v = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    s.put_embedding(t.id, v, "clap")
    assert np.allclose(s.embedding(t.id), v)
    assert set(s.embeddings([t.id, 999])) == {t.id}
    assert s.embedding(999) is None


def test_sessions_and_positions(s):
    t1 = s.upsert_track("A", "1"); t2 = s.upsert_track("B", "2"); t3 = s.upsert_track("C", "3")
    t0 = 1_700_000_000.0
    s.add_event("play", t1.id, "spotify", t=t0)
    s.add_event("play", t2.id, "spotify", t=t0 + 200)
    s.add_event("play", t3.id, "spotify", t=t0 + 3 * 3600)  # new session
    ev = s.events(kinds=("play",))
    assert [e["position_in_session"] for e in ev] == [0, 1, 0]
    assert ev[0]["session_id"] == ev[1]["session_id"] != ev[2]["session_id"]
    assert ev[1]["prev_track_id"] == t1.id


def test_ctx_json_roundtrip(s):
    t = s.upsert_track("A", "1")
    s.add_event("play", t.id, "spotify", ctx={"home_distance": 0.31, "knobs": {"home_pull": 1.0}})
    assert s.events()[0]["ctx"]["knobs"]["home_pull"] == 1.0


def test_seen_covers_plays_kicks_and_picks(s):
    t = s.upsert_track("Tinariwen", "Amassakoul")
    assert not s.seen("Tinariwen", "Amassakoul")
    s.add_event("play", t.id, "spotify")
    assert s.seen("tinariwen", "amassakoul")
    k = s.upsert_track("Ed Motta", "Manuel")
    s.add_kick(strength="boot", magnitude=0.9, target_rel=1.3, direction="brazilian soul", why="", track_id=k.id, distance=0.5,
               rel=1.2, band="boot", dose=5, pre_state=np.zeros(4), kick_vec=np.ones(4))
    assert s.seen("Ed Motta", "Manuel")
    assert not s.seen("Nobody", "Nothing")


def test_context_queries(s):
    now = time.time()
    fela = s.upsert_track("Fela Kuti", "Zombie"); fela2 = s.upsert_track("Fela Kuti", "Water No Get Enemy")
    st = s.upsert_track("Stereolab", "Ping Pong"); old = s.upsert_track("Old Band", "Old Song")
    s.add_event("play", old.id, "spotify", t=now - 90 * 86400)
    s.add_event("play", old.id, "spotify", t=now - 89 * 86400)
    s.add_event("play", old.id, "spotify", t=now - 88 * 86400)
    s.add_event("play", fela.id, "spotify", t=now - 3000)
    s.add_event("play", fela2.id, "kick", t=now - 2000)
    s.add_event("partial", st.id, "spotify", t=now - 1000, completion=0.4)
    s.add_event("skip", st.id, "spotify", t=now - 500, skip_at_s=20)
    s.add_event("love", fela.id, "user", t=now - 100)

    r = s.recent(2)
    assert [x["title"] for x in r] == ["Ping Pong", "Ping Pong"] and r[0]["kind"] == "skip"
    assert s.top_artists(days=30, n=5)[0] == ("Fela Kuti", 2)
    assert s.top_artists(n=1)[0] == ("Old Band", 3)
    assert s.loved() == ["Fela Kuti — Zombie"]
    assert s.rejected() == ["Stereolab — Ping Pong"]

    k = s.upsert_track("Ed Motta", "Manuel")
    kid = s.add_kick(strength="kick", magnitude=0.5, target_rel=0.75, direction="brazilian soul", why="", track_id=k.id, distance=0.4,
                     rel=0.7, band="kick", dose=3, pre_state=None, kick_vec=None, t=now - 50)
    s.add_event("kick", k.id, "kick", kick_id=kid, t=now - 50)
    s.add_event("play", k.id, "kick", t=now - 49)              # the forced follow-through: not counted
    after = s.upsert_track("Jorge Ben Jor", "Taj Mahal")
    s.add_event("play", after.id, "spotify", t=now - 10)
    assert s.directions() == ["brazilian soul"] and s.kicked_artists() == ["Ed Motta"]
    since = s.plays_since_kick(kid)
    assert [p["artist"] for p in since] == ["Jorge Ben Jor"]
    s.update_kick(kid, followed=0.7, verdict="followed", n_since=1)
    assert s.last_kick()["verdict"] == "followed"
    with pytest.raises(ValueError):
        s.update_kick(kid, distance=0.1)


def test_candidates_lifecycle(s):
    ids = s.add_candidates("set-1", [
        {"reach": "near", "direction": "d1", "artist": "A", "title": "1", "why": "", "spotify_uri": "spotify:track:x"},
        {"reach": "far", "direction": "d2", "artist": "B", "title": "2", "why": ""},
        {"reach": "far", "direction": "d3", "artist": "C", "title": "3", "why": "", "rejected_reason": "seen"},
    ])
    t = s.upsert_track("A", "1")
    s.update_candidate(ids[0], track_id=t.id, distance=0.2, rel=0.3, band="tap")
    usable = s.latest_candidate_set()
    assert [c["artist"] for c in usable] == ["A"]          # B unresolved, C rejected
    s.update_candidate(ids[0], chosen=1)
    assert s.latest_candidate_set() == []
    assert len(s.candidate_set("set-1")) == 3
    assert s.counts()["candidates"] == 3


def test_follow_through_sets_are_never_the_latest_pool(s):
    ids = s.add_candidates("pool-1", [{"reach": "far", "direction": "d", "artist": "A", "title": "1", "why": ""}])
    s.update_candidate(ids[0], track_id=s.upsert_track("A", "1").id)
    fids = s.add_candidates("follow-1", [{"reach": "adjacent", "direction": "d", "artist": "F", "title": "2", "why": ""}], purpose="follow")
    s.update_candidate(fids[0], track_id=s.upsert_track("F", "2").id)
    assert [c["artist"] for c in s.latest_candidate_set()] == ["A"]              # the newer follow set is not a pool
    assert [c["artist"] for c in s.latest_candidate_set(purpose="follow")] == ["F"]


def test_migration_adds_purpose_and_tags_old_follow_sets(tmp_path):
    path = tmp_path / "old.db"
    Store(str(path)).close()
    db = sqlite3.connect(path)
    db.executescript("ALTER TABLE candidates DROP COLUMN purpose;"
                     "INSERT INTO candidates(t,set_id,for_track_id,reach,artist,title) VALUES (1,'f',NULL,'adjacent','X','1'),"
                     " (1,'f',NULL,'adjacent','Y','2'), (2,'p',NULL,'near','Z','3'), (2,'p',NULL,'far','W','4');")
    db.commit(); db.close()
    st = Store(str(path))
    rows = {r["set_id"]: r["purpose"] for r in st._all("SELECT DISTINCT set_id, purpose FROM candidates")}
    assert rows == {"f": "follow", "p": "pool"}


def test_persists_to_disk(tmp_path):
    p = tmp_path / "x" / "spotkick.db"
    a = Store(p); t = a.upsert_track("A", "1"); a.add_event("play", t.id, "spotify"); a.close()
    b = Store(p)
    assert b.counts() == {"tracks": 1, "embeddings": 0, "events": 1, "kicks": 0, "candidates": 0}
    assert b.get_config("nope", "dflt") == "dflt"
    b.set_config("llm_backend", "local"); assert b.get_config("llm_backend") == "local"
    b.set_profile("home", b"\x00\x01"); assert b.get_profile("home") == b"\x00\x01"
