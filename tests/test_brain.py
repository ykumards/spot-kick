import json
import time

import pytest

from spotkick.brain import propose as PR
from spotkick.brain.codex import Codex
from spotkick.brain.llm import BrainError, make_backend
from spotkick.brain.openai_compat import OpenAICompat
from spotkick.brain.prompts import CANDIDATES_SCHEMA, Context, candidates_prompt
from spotkick.config import Config
from spotkick.mind.store import Store


def seeded_store() -> Store:
    s = Store(":memory:")
    now = time.time()
    fela = s.upsert_track("Fela Kuti", "Zombie"); st = s.upsert_track("Stereolab", "Ping Pong")
    ed = s.upsert_track("Ed Motta", "Manuel"); old = s.upsert_track("Old Band", "Old Song")
    for i in range(3):
        s.add_event("play", old.id, "spotify", t=now - (90 - i) * 86400)
    s.add_event("play", fela.id, "spotify", t=now - 3000)
    s.add_event("skip", st.id, "spotify", t=now - 500, skip_at_s=20)
    s.add_event("love", fela.id, "user", t=now - 100)
    kid = s.add_kick(strength="kick", magnitude=0.5, target_rel=0.75, direction="brazilian soul", why="", track_id=ed.id, distance=0.4,
                     rel=0.7, band="kick", dose=3, pre_state=None, kick_vec=None, t=now - 50)
    s.add_event("kick", ed.id, "kick", kick_id=kid, t=now - 50)
    s.add_event("play", ed.id, "kick", kick_id=kid, t=now - 49)
    return s


# ------------------------------------------------------------------ prompts
def test_context_lines_are_capped_and_tagged():
    ctx = Context.from_store(seeded_store(), taste=["afrobeat & highlife", "krautrock-adjacent pop"])
    lines = ctx.lines()
    text = "\n".join(lines)
    assert lines[0] == "Last plays, most recent first:"
    assert "- Ed Motta — Manuel [kick]" in lines and "- Stereolab — Ping Pong (skipped)" in lines
    assert "Most played artists, last 30 days: Ed Motta (1), Fela Kuti (1)" in text
    assert "all time: Old Band (3)" in text
    assert "Loved: Fela Kuti — Zombie" in text and "not this vein: Stereolab — Ping Pong" in text
    assert "Directions already kicked toward (choose different ones): brazilian soul" in text
    assert "Artists already kicked to, avoid: Ed Motta" in text and "afrobeat & highlife" in text
    assert len(lines) <= 22


def test_prompt_size_does_not_grow_with_history():
    s = seeded_store()
    small = len(candidates_prompt(Context.from_store(s)))
    now = time.time()
    for i in range(400):
        t = s.upsert_track(f"Artist {i % 37}", f"Song {i}")
        s.add_event("play", t.id, "spotify", t=now - 40000 + i * 60)
    big = len(candidates_prompt(Context.from_store(s)))
    assert big < small * 2 and s.counts()["events"] > 400


def test_prompt_variants():
    ctx = Context.from_store(seeded_store())
    p = candidates_prompt(ctx, n=6, dig=2)
    assert "2 labelled 'near'" in p and "Go deep" in p and "spotify:track:<22 chars>" in p
    p2 = candidates_prompt(ctx, n=4, dig=0, direction_hint="brazilian soul", rejects=["A — B"])
    assert 'ONE direction: "brazilian soul"' in p2 and "rejected (already known" in p2 and "Go deep" not in p2
    assert CANDIDATES_SCHEMA["properties"]["candidates"]["items"]["required"] == ["reach", "direction", "artist", "title", "why", "spotify_uri"]


# ------------------------------------------------------------------ propose + dedup
class FakeBackend:
    name = "fake"

    def __init__(self, *sets):
        self.sets, self.prompts = list(sets), []

    def complete_json(self, prompt, schema, *, timeout=240):
        self.prompts.append(prompt)
        return {"candidates": self.sets.pop(0)}

    def search_uri(self, artist, title):
        return None


def cand(artist, title, reach="far", uri="", direction="d", why="w"):
    return {"reach": reach, "direction": direction, "artist": artist, "title": title, "why": why, "spotify_uri": uri}


def test_propose_rejects_known_and_duplicates_and_retries():
    s = seeded_store()
    b = FakeBackend(
        [cand("Fela Kuti", "Zombie"), cand("Ed Motta", "Manuel"), cand("Tinariwen", "Amassakoul", uri="spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"),
         cand("tinariwen", "amassakoul"), cand("", "x")],
        [cand("Azymuth", "Linha do Horizonte", why="[see](http://x) groove"), cand("Novos Baianos", "Preta Pretinha")],
    )
    out = PR.propose(b, s, min_fresh=3)
    reasons = {(c.artist, c.title): c.rejected_reason for c in out}
    assert reasons[("Fela Kuti", "Zombie")] == "already known" and reasons[("Ed Motta", "Manuel")] == "already known"
    assert reasons[("Tinariwen", "Amassakoul")] is None and reasons[("tinariwen", "amassakoul")] == "duplicate in set"
    assert reasons[("Azymuth", "Linha do Horizonte")] is None and reasons[("Novos Baianos", "Preta Pretinha")] is None
    assert len(out) == 6 and len(b.prompts) == 2 and "Fela Kuti — Zombie" in b.prompts[1]
    az = next(c for c in out if c.artist == "Azymuth")
    assert az.why == "see groove" and az.spotify_uri is None
    assert next(c for c in out if c.artist == "Tinariwen").spotify_uri == "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"


def test_propose_no_retry_when_enough():
    b = FakeBackend([cand("A", "1"), cand("B", "2"), cand("C", "3")])
    out = PR.propose(b, Store(":memory:"), min_fresh=3)
    assert len(out) == 3 and len(b.prompts) == 1


# ------------------------------------------------------------------ backends
def test_codex_backend_builds_command_and_reads_output(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w") as f:
            json.dump({"candidates": []}, f)

        class R:
            returncode, stderr = 0, ""
        return R()

    import spotkick.brain.codex as C
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    got = Codex(model="m", reasoning="low").complete_json("hi", CANDIDATES_SCHEMA)
    assert got == {"candidates": []}
    c = seen["cmd"]
    assert c[:2] == ["codex", "exec"] and "--output-schema" in c and c[c.index("-m") + 1] == "m" and c[-1] == "hi"

    def fail_run(cmd, **kw):
        class R:
            returncode, stderr = 1, "boom"
        return R()
    monkeypatch.setattr(C.subprocess, "run", fail_run)
    with pytest.raises(BrainError):
        Codex().complete_json("hi", CANDIDATES_SCHEMA)


def test_openai_compat_posts_json_schema():
    posted = {}

    class Sess:
        def post(self, url, json=None, timeout=None, headers=None):
            posted.update(url=url, body=json, headers=headers)

            class R:
                status_code = 200

                def json(self):
                    return {"choices": [{"message": {"content": '{"candidates": [{"artist": "A"}]}'}}]}
            return R()

    b = OpenAICompat(model="gpt-x", api_key="k", session=Sess())
    assert b.name == "openai"
    assert b.complete_json("p", CANDIDATES_SCHEMA)["candidates"][0]["artist"] == "A"
    assert posted["url"] == "https://api.openai.com/v1/chat/completions" and posted["body"]["reasoning_effort"] == "low"
    assert posted["body"]["response_format"]["json_schema"]["schema"] == CANDIDATES_SCHEMA
    assert posted["headers"]["Authorization"] == "Bearer k"

    local = OpenAICompat(model="qwen", base_url="http://127.0.0.1:8080/v1", session=Sess())
    assert local.name == "local" and local.api_key == "local"
    local.complete_json("p", CANDIDATES_SCHEMA)
    assert posted["url"] == "http://127.0.0.1:8080/v1/chat/completions" and "reasoning_effort" not in posted["body"]
    assert local.search_uri("a", "t") is None


def test_make_backend_from_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert make_backend(Config(llm_backend="codex")).name == "codex"
    assert make_backend(Config(llm_backend="openai")).name == "openai"
    assert make_backend(Config(llm_backend="local")).base_url == "http://127.0.0.1:8080/v1"
    with pytest.raises(BrainError):
        make_backend(Config(llm_backend="claude"))
