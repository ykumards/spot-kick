import json
import time

import pytest

from spotkick.brain import cli
from spotkick.brain.claude import ClaudeCode
from spotkick.brain.codex import Codex
from spotkick.brain.llm import BrainError, make_backend
from spotkick.brain.prompts import CANDIDATES_SCHEMA, Context, candidates_prompt
from spotkick.brain.propose import propose
from spotkick.config import Config
from spotkick.memory.store import Store

DAY_S = 86400
GOOD_URI = "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"


def seeded_store() -> Store:
    store = Store(":memory:")
    now = time.time()
    fela = store.upsert_track("Fela Kuti", "Zombie")
    stereolab = store.upsert_track("Stereolab", "Ping Pong")
    ed_motta = store.upsert_track("Ed Motta", "Manuel")
    old_band = store.upsert_track("Old Band", "Old Song")
    for days_back in (90, 89, 88):
        store.add_event("play", old_band.id, "spotify", t=now - days_back * DAY_S)
    store.add_event("play", fela.id, "spotify", t=now - 3000)
    store.add_event("skip", stereolab.id, "spotify", t=now - 500, skip_at_s=20)
    store.add_event("love", fela.id, "user", t=now - 100)
    kick_id = store.add_kick(
        strength="kick",
        magnitude=0.5,
        target_rel=0.75,
        direction="brazilian soul",
        why="",
        track_id=ed_motta.id,
        distance=0.4,
        rel=0.7,
        band="kick",
        pre_state=None,
        kick_vec=None,
        t=now - 50,
    )
    store.add_event("kick", ed_motta.id, "kick", kick_id=kick_id, t=now - 50)
    store.add_event("play", ed_motta.id, "kick", kick_id=kick_id, t=now - 49)
    return store


# ------------------------------------------------------------------ prompts
def test_context_lines_are_capped_and_tagged():
    context = Context.from_store(seeded_store())
    lines = context.lines()
    text = "\n".join(lines)
    assert lines[0] == "Last plays, most recent first:"
    assert "- Ed Motta — Manuel [kick]" in lines
    assert "- Stereolab — Ping Pong (skipped)" in lines
    assert "Most played artists, last 30 days: Ed Motta (1), Fela Kuti (1)" in text
    assert "all time: Old Band (3)" in text
    assert "Loved: Fela Kuti — Zombie" in text
    assert "not this vein: Stereolab — Ping Pong" in text
    assert "Directions already kicked toward (choose different ones): brazilian soul" in text
    assert "Artists already kicked to, avoid: Ed Motta" in text
    assert len(lines) <= 22


def test_prompt_size_does_not_grow_with_history():
    store = seeded_store()
    small = len(candidates_prompt(Context.from_store(store)))
    now = time.time()
    for index in range(400):
        track = store.upsert_track(f"Artist {index % 37}", f"Song {index}")
        store.add_event("play", track.id, "spotify", t=now - 40000 + index * 60)
    big = len(candidates_prompt(Context.from_store(store)))
    assert big < small * 2
    assert store.counts()["events"] > 400


def test_prompt_variants():
    context = Context.from_store(seeded_store())
    spread_prompt = candidates_prompt(context, n=6)
    assert "2 labelled 'near'" in spread_prompt
    assert "go deep" in spread_prompt                                # obscurity comes from the reach text
    assert "spotify" not in spread_prompt.split("For each:")[1]       # the brain is never asked for ids
    far_prompt = candidates_prompt(context, n=4, reach="far")
    assert "Propose 4 real songs, each in a DIFFERENT direction, all labelled 'far'" in far_prompt
    assert "labelled 'near'" not in far_prompt
    assert "did not land" not in far_prompt
    misses = [{"artist": "Konono No. 1", "title": "Mama Lissanga", "band": "tap"}]
    corrected_prompt = candidates_prompt(context, n=6, reach="far", misses=misses)
    assert "Konono No. 1 — Mama Lissanga measured as a small step" in corrected_prompt
    assert "do not propose them again" in corrected_prompt
    retry_prompt = candidates_prompt(context, n=4, rejects=["A — B"])
    assert "rejected (already known" in retry_prompt
    leaning_prompt = candidates_prompt(context, n=6, lean="  melancholic,\n  Portuguese ")
    assert 'stay inside it: "melancholic, Portuguese"' in leaning_prompt
    assert "the lean wins" in leaning_prompt
    assert "lean" not in spread_prompt


class FakeBackend:
    name = "fake"

    def __init__(self, *candidate_sets):
        self.sets = list(candidate_sets)
        self.prompts = []

    def complete_json(self, prompt, schema, *, timeout=240):
        self.prompts.append(prompt)
        return {"candidates": self.sets.pop(0)}


def candidate_row(artist, title, reach="far", direction="d", why="w"):
    return {"reach": reach, "direction": direction, "artist": artist, "title": title, "why": why}


def test_propose_rejects_known_and_duplicates_and_retries():
    store = seeded_store()
    first_set = [
        candidate_row("Fela Kuti", "Zombie"),
        candidate_row("Ed Motta", "Manuel"),
        candidate_row("Tinariwen", "Amassakoul"),
        candidate_row("tinariwen", "amassakoul"),
        candidate_row("", "x"),
    ]
    second_set = [
        candidate_row("Azymuth", "Linha do Horizonte", why="[see](http://x) groove"),
        candidate_row("Novos Baianos", "Preta Pretinha"),
    ]
    backend = FakeBackend(first_set, second_set)
    candidates = propose(backend, store, min_fresh=3)
    reasons = {(candidate.artist, candidate.title): candidate.rejected_reason for candidate in candidates}
    assert reasons[("Fela Kuti", "Zombie")] == "already known"
    assert reasons[("Ed Motta", "Manuel")] == "already known"
    assert reasons[("Tinariwen", "Amassakoul")] is None
    assert reasons[("tinariwen", "amassakoul")] == "duplicate in set"
    assert reasons[("Azymuth", "Linha do Horizonte")] is None
    assert reasons[("Novos Baianos", "Preta Pretinha")] is None
    assert len(candidates) == 6
    assert len(backend.prompts) == 2
    assert "Fela Kuti — Zombie" in backend.prompts[1]
    azymuth = next(candidate for candidate in candidates if candidate.artist == "Azymuth")
    assert azymuth.why == "see groove"


def test_propose_no_retry_when_enough():
    backend = FakeBackend([candidate_row("A", "1"), candidate_row("B", "2"), candidate_row("C", "3")])
    candidates = propose(backend, Store(":memory:"), min_fresh=3)
    assert len(candidates) == 3
    assert len(backend.prompts) == 1


# ------------------------------------------------------------------ backends
class FakeCompletedProcess:
    def __init__(self, returncode, stderr, stdout=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_codex_backend_builds_command_and_reads_output(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        output_path = command[command.index("-o") + 1]
        with open(output_path, "w") as output_file:
            json.dump({"candidates": []}, output_file)
        return FakeCompletedProcess(returncode=0, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    got = Codex(model="m", reasoning="low").complete_json("hi", CANDIDATES_SCHEMA)
    assert got == {"candidates": []}
    command = seen["command"]
    assert command[:2] == ["codex", "exec"]
    assert "--output-schema" in command
    assert command[command.index("-m") + 1] == "m"
    assert command[-1] == "hi"

    def fail_run(command, **kwargs):
        return FakeCompletedProcess(returncode=1, stderr="boom")

    monkeypatch.setattr(cli.subprocess, "run", fail_run)
    with pytest.raises(BrainError):
        Codex().complete_json("hi", CANDIDATES_SCHEMA)


def test_make_backend_from_config():
    assert make_backend(Config()).name == "codex"
    assert make_backend(Config(llm_backend="claude")).name == "claude"
    assert make_backend(Config(llm_backend="claude", claude_model="haiku")).model == "haiku"
    with pytest.raises(BrainError):
        make_backend(Config(llm_backend="vibe"))


class FakeStdoutProcess:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def claude_envelope(**fields) -> str:
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, **fields})


def test_claude_backend_builds_command_and_reads_structured_output(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["cwd"] = kwargs.get("cwd")
        return FakeStdoutProcess(0, claude_envelope(structured_output={"candidates": []}, result="{}"))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    got = ClaudeCode(model="haiku").complete_json("hi", CANDIDATES_SCHEMA)
    assert got == {"candidates": []}
    command = seen["command"]
    assert command[:2] == ["claude", "-p"]
    assert command[command.index("--model") + 1] == "haiku"
    assert command[command.index("--tools") + 1] == ""                   # proposals get no tools at all
    assert command[command.index("--tools") + 2] == "--max-turns"        # the variadic --tools must not eat the prompt
    assert json.loads(command[command.index("--json-schema") + 1]) == CANDIDATES_SCHEMA
    assert "--no-session-persistence" in command
    assert command[-1] == "hi"
    assert seen["cwd"] is not None                                        # never run from the user's project


def test_claude_backend_errors_are_brain_errors(monkeypatch):
    def max_turns(command, **kwargs):
        return FakeStdoutProcess(0, claude_envelope(is_error=True, errors=["Reached maximum number of turns (1)"]))

    monkeypatch.setattr(cli.subprocess, "run", max_turns)
    with pytest.raises(BrainError, match="maximum number of turns"):
        ClaudeCode().complete_json("hi", CANDIDATES_SCHEMA)

    def no_structure(command, **kwargs):
        return FakeStdoutProcess(0, claude_envelope(result="just prose"))

    monkeypatch.setattr(cli.subprocess, "run", no_structure)
    with pytest.raises(BrainError, match="no structured output"):
        ClaudeCode().complete_json("hi", CANDIDATES_SCHEMA)

    def not_json(command, **kwargs):
        return FakeStdoutProcess(0, "Please run /login first")

    monkeypatch.setattr(cli.subprocess, "run", not_json)
    with pytest.raises(BrainError, match="non-JSON"):
        ClaudeCode().complete_json("hi", CANDIDATES_SCHEMA)

    def missing(command, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(cli.subprocess, "run", missing)
    with pytest.raises(BrainError, match="claude CLI not found"):
        ClaudeCode().complete_json("hi", CANDIDATES_SCHEMA)



def test_make_backend_carries_each_brain_setting():
    hosted = make_backend(Config(llm_model="gpt-5.6-luna", llm_reasoning="high"))
    assert isinstance(hosted, Codex)
    assert hosted.model == "gpt-5.6-luna"
    assert hosted.reasoning == "high"
    assert "--oss" not in hosted.base_command()
    assert 'model_reasoning_effort="high"' in hosted.base_command()


# ------------------------------------------------------------------ config
def test_save_setting_rewrites_one_line_and_keeps_the_rest(tmp_path):
    from spotkick import config

    path = tmp_path / "config.toml"
    path.write_text('# my settings\nllm_backend = "codex"\nlean = "jazz"\n')
    config.save_setting("llm_backend", "claude", path)
    assert path.read_text() == '# my settings\nllm_backend = "claude"\nlean = "jazz"\n'
    assert config.load(path).llm_backend == "claude"
    assert config.load(path).lean == "jazz"

    config.save_setting("claude_model", "haiku", path)
    assert config.load(path).claude_model == "haiku"

    fresh = tmp_path / "new" / "config.toml"
    config.save_setting("llm_backend", "claude", fresh)                  # creates the file and its directory
    assert config.load(fresh).llm_backend == "claude"
