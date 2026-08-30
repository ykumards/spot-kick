"""The parity fixtures are the contract a port is checked against, so they must keep matching this implementation.

If one of these fails, either the Python behaviour changed on purpose (re-run
`scripts/export_parity_fixtures.py` and tell the port) or it changed by accident (fix the code).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from spotkick.brain.prompts import Context, candidates_prompt
from spotkick.ears import clap, features
from spotkick.kick import bands
from spotkick.mind.store import Store

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures" / "parity"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Imported after the path insert above; the exporter owns the wave specs these tests re-render.
from export_parity_fixtures import render_wave

pytestmark = pytest.mark.skipif(not (FIXTURES / "manifest.json").exists(),
                                reason="parity fixtures not exported")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((FIXTURES / "manifest.json").read_text())


def read_f32(entry: dict) -> np.ndarray:
    raw = np.frombuffer((FIXTURES / entry["file"]).read_bytes(), dtype="<f4")
    return raw.reshape(entry["shape"])


def test_log_mel_and_features_match_the_fixtures(manifest):
    mel_filters = read_f32(manifest["mel_filters"])
    tolerance = manifest["tolerances"]["logmel_max_abs_db"]
    for wave_spec, logmel_entry, features_entry in zip(manifest["waves"], manifest["logmel"], manifest["features"]):
        wave = render_wave(wave_spec)
        assert np.abs(features.log_mel(wave, mel_filters) - read_f32(logmel_entry)).max() <= tolerance
        assert np.abs(features.input_features(wave, mel_filters) - read_f32(features_entry)).max() <= tolerance


@pytest.mark.skipif(not clap.model_present(), reason="CLAP model not installed")
def test_embeddings_match_the_fixtures(manifest):
    embedder = clap.Embedder()
    tolerance = manifest["tolerances"]["embedding_max_cosine_distance"]
    for wave_spec, embedding_entry in zip(manifest["waves"], manifest["embeddings"]):
        embedded = embedder.embed_audio(render_wave(wave_spec))
        assert float(1.0 - embedded @ read_f32(embedding_entry)) <= tolerance


def test_bands_arithmetic_matches_the_fixtures():
    cases = json.loads((FIXTURES / "bands.json").read_text())
    history = np.array(cases["history"], dtype=np.float32)
    candidates = np.array(cases["candidates"], dtype=np.float32)

    state = bands.ListenerState(alpha=cases["alpha"])
    for embedding in history:
        state.update(embedding)
    assert state.vector is not None
    assert np.allclose(state.vector, cases["state_vector"], atol=1e-6)

    typical_step, far = state.scale()
    assert typical_step == pytest.approx(cases["typical_step"])
    assert far == pytest.approx(cases["far"])

    for item, expected in zip(bands.measure(state, list(candidates)), cases["measured"]):
        assert item.distance == pytest.approx(expected["distance"])
        assert item.rel == pytest.approx(expected["rel"])
        assert item.band == expected["band"]

    for expected in cases["magnitudes"]:
        assert bands.strength_for(expected["magnitude"]) == expected["strength"]
        assert bands.target_for(expected["magnitude"]) == pytest.approx(expected["target_rel"])

    followed = cases["followed"]
    value = bands.followed(np.array(followed["pre"], dtype=np.float32),
                           np.array(followed["kick"], dtype=np.float32),
                           np.array(followed["now"], dtype=np.float32))
    assert value == pytest.approx(followed["value"])

    for expected in cases["verdicts"]:
        assert bands.verdict(expected["followed"], expected["n_since"]) == expected["verdict"]


def test_prompts_match_the_fixtures(manifest):
    store = Store(":memory:")
    plays = [
        ("Soft Machine", "Hazard Profile Pt.1", "play"),
        ("Milton Nascimento", "Clube da Esquina No 2", "play"),
        ("The Durutti Column", "Sketch for Summer", "skip"),
        ("Alice Coltrane", "Journey in Satchidananda", "play"),
        ("Can", "Vitamin C", "partial"),
    ]
    base_time = 1_700_000_000.0
    for index, (artist, title, kind) in enumerate(plays):
        track = store.upsert_track(artist, title)
        store.add_event(kind, track.id, "spotify", t=base_time + index * 300)

    context = Context.from_store(store)
    for entry in manifest["prompts"]:
        expected = (FIXTURES / entry["file"]).read_text()
        assert candidates_prompt(context, **entry["kwargs"]) == expected
    store.close()
