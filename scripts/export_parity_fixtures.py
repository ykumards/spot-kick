"""Freeze what the Python implementation computes, so a port can be checked against it instead of against a guess.

Writes `fixtures/parity/` — every file the Swift port's tests read:

    manifest.json          what each file is, how to rebuild the test audio, the tolerances, the git commit
    logmel_*.f32           log_mel(wave) for each test wave, (frames, 64)
    features_*.f32         input_features(wave), (n_clips, 1, 1001, 64)
    embedding_*.f32        Embedder.embed_audio(wave), (512,) — only when the ONNX model is installed
    bands.json             listener-state / scale / rel / band / target / followed / verdict cases
    context.json           the Context those prompts were built from, so a port renders from identical input
    prompt_*.txt           candidates_prompt() for that context, byte-for-byte

The test audio is *not* shipped: the manifest describes each wave as a sum of analytic components (sine, chirp, and
a plain LCG noise whose arithmetic is identical in any language), so a port generates the same input samples and only
the outputs need to be stored. Raw `.f32` is little-endian float32 with no header; shapes are in the manifest.
Run from the repo root:

    .venv/bin/python scripts/export_parity_fixtures.py
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spotkick.brain.prompts import Context, candidates_prompt
from spotkick.ears import clap, features
from spotkick.kick import bands
from spotkick.mind.store import Store

OUT_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "parity"
SAMPLE_RATE = features.SR

# What a port is allowed to differ by. Nothing here can be exact: numpy sums float32 dot products in a different
# order than Accelerate does, and vDSP's FFT and numpy's are both correct without being bit-identical. The
# tolerances are set where the difference stops being noise and starts being a bug — several orders of magnitude
# below anything that could move a band boundary or a verdict.
TOLERANCES = {
    "logmel_max_abs_db": 1e-3,          # replaced below by the measured sensitivity when that is larger
    "features_max_abs_db": 1e-3,
    "embedding_max_cosine_distance": 1e-4,
    "bands_state_max_abs": 1e-6,        # the EWMA state vector, float32 either way
    "bands_distance_max_abs": 1e-6,     # one cosine distance
    "bands_scale_max_abs": 1e-6,        # median and 95th percentile of pairwise distances
    "bands_rel_max_abs": 1e-5,          # (d − step) / (far − step): a ~0.1 denominator amplifies the above
    "bands_exact_max_abs": 1e-9,        # target_for, acceptance, followed: plain arithmetic on given numbers
}


# The test audio, described rather than stored. Every component is exactly reproducible in any language: `sine` and
# `chirp` are libm arithmetic, and `noise` is the textbook 32-bit linear congruential generator below rather than
# numpy's PCG64, which a port could not reproduce.
LCG_MULTIPLIER = 1_664_525
LCG_INCREMENT = 1_013_904_223
LCG_MODULUS = 2**32

WAVE_SPECS = [
    {
        "name": "long",
        "seconds": 15.0,
        "description": "15 s, three deterministic crops",
        "components": [
            {"kind": "sine", "hz": 440.0, "amplitude": 0.5},
            {"kind": "chirp", "start_hz": 80.0, "end_hz": 12000.0, "amplitude": 0.3},
            {"kind": "noise", "seed": 7, "amplitude": 0.1},
        ],
    },
    {
        # The noise floor is deliberate: a bare chirp leaves mel bins near −100 dB, where log10 turns a one-bit
        # input difference into a tenth of a decibel and the parity tolerance would have to be uselessly loose.
        "name": "exact",
        "seconds": 10.0,
        "description": "exactly 10 s, one middle crop",
        "components": [
            {"kind": "chirp", "start_hz": 50.0, "end_hz": 14000.0, "amplitude": 0.5},
            {"kind": "noise", "seed": 5, "amplitude": 0.02},
        ],
    },
    {
        "name": "short",
        "seconds": 2.5,
        "description": "2.5 s, repeat-padded to 10 s",
        "components": [
            {"kind": "sine", "hz": 220.0, "amplitude": 0.5},
            {"kind": "noise", "seed": 3, "amplitude": 0.2},
        ],
    },
]


@dataclass(frozen=True)
class Wave:
    name: str
    samples: np.ndarray
    description: str


def lcg_noise(count: int, seed: int) -> np.ndarray:
    """Uniform noise in [-1, 1) from a 32-bit LCG: `state = (a * state + c) mod 2**32`, taken in that order so a
    port that runs the same recurrence gets the same samples."""
    samples = np.empty(count, dtype=np.float64)
    state = seed % LCG_MODULUS
    for index in range(count):
        state = (LCG_MULTIPLIER * state + LCG_INCREMENT) % LCG_MODULUS
        samples[index] = state / LCG_MODULUS * 2.0 - 1.0
    return samples


def render_wave(spec: dict) -> np.ndarray:
    """A wave spec from the manifest → mono float32 samples at 48 kHz."""
    count = int(SAMPLE_RATE * spec["seconds"])
    time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    samples = np.zeros(count, dtype=np.float64)
    for component in spec["components"]:
        amplitude = component["amplitude"]
        if component["kind"] == "sine":
            samples += amplitude * np.sin(2 * np.pi * component["hz"] * time)
        elif component["kind"] == "chirp":
            start_hz, end_hz = component["start_hz"], component["end_hz"]
            sweep = start_hz * time + (end_hz - start_hz) * time**2 / (2 * spec["seconds"])
            samples += amplitude * np.sin(2 * np.pi * sweep)
        elif component["kind"] == "noise":
            samples += amplitude * lcg_noise(count, component["seed"])
        else:
            raise ValueError(f"unknown wave component {component['kind']!r}")
    return samples.astype(np.float32)


def build_waves() -> list[Wave]:
    """Three shapes that exercise different paths: a long clip that gets three crops, an exactly-10 s clip that
    gets one, and a short clip that must be repeat-padded."""
    return [Wave(spec["name"], render_wave(spec), spec["description"]) for spec in WAVE_SPECS]


def ulp_sensitivity(wave: np.ndarray, mel_filters: np.ndarray, seed: int = 0) -> float:
    """How far the log-mel output moves when every input sample is nudged by one float32 ULP.

    A port regenerates the test audio from the manifest rather than reading it, and two correct `sin`
    implementations may differ in the last bit. In near-silent mel bins (a pure chirp reaches −100 dB) that last bit
    is worth far more than the front end's own arithmetic noise, so the parity tolerance is set from this
    measurement instead of from taste.
    """
    generator = np.random.default_rng(seed)
    base = features.log_mel(wave, mel_filters)
    one_ulp = np.nextafter(wave, np.inf, dtype=np.float32) - wave
    signs = generator.choice([-1.0, 1.0], size=wave.shape)
    nudged = (wave + one_ulp * signs).astype(np.float32)
    return float(np.abs(features.log_mel(nudged, mel_filters) - base).max())


def write_f32(path: Path, array: np.ndarray) -> dict:
    data = np.ascontiguousarray(array, dtype="<f4")
    path.write_bytes(data.tobytes())
    return {"file": path.name, "shape": list(data.shape), "dtype": "float32-le"}


def bands_cases() -> dict:
    """The kick's arithmetic on fixed inputs: no audio, no model, pure numbers a port must match exactly."""
    generator = np.random.default_rng(11)
    history = generator.standard_normal((8, 16)).astype(np.float32)
    history /= np.linalg.norm(history, axis=1, keepdims=True)

    state = bands.ListenerState(alpha=0.7)
    for embedding in history:
        state.update(embedding)

    candidates = generator.standard_normal((5, 16)).astype(np.float32)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    measured = bands.measure(state, list(candidates))
    typical_step, far = state.scale()

    magnitudes = [0.0, 0.1, 0.165, 0.4, 0.5, 0.7, 0.83, 0.95, 1.0]
    pre, kick_vector, now = history[0], candidates[0], history[3]

    return {
        "history": [row.tolist() for row in history],
        "candidates": [row.tolist() for row in candidates],
        "alpha": 0.7,
        "state_vector": state.vector.tolist() if state.vector is not None else None,
        "typical_step": typical_step,
        "far": far,
        "measured": [
            {"index": item.index, "distance": item.distance, "rel": item.rel, "band": item.band,
             "acceptance": item.acceptance}
            for item in measured
        ],
        "magnitudes": [
            {"magnitude": magnitude, "strength": bands.strength_for(magnitude),
             "target_rel": bands.target_for(magnitude), "acceptance": bands.acceptance(bands.target_for(magnitude))}
            for magnitude in magnitudes
        ],
        "followed": {
            "pre": pre.tolist(), "kick": kick_vector.tolist(), "now": now.tolist(),
            "value": bands.followed(pre, kick_vector, now),
        },
        "verdicts": [
            {"followed": value, "n_since": n_since, "verdict": bands.verdict(value, n_since)}
            for value, n_since in [(0.0, 0), (0.0, 2), (0.1, 3), (0.4, 3), (0.7, 3), (-0.5, 4), (1.4, 5)]
        ],
    }


def prompt_fixtures(out_dir: Path) -> list[dict]:
    """A fixed listening history through the real store, then the exact prompt text the brain would receive."""
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

    written = []
    context = Context.from_store(store)
    # The context is dumped too: a port should be checked on the prompt it renders, not on whether it reimplemented
    # the store queries that fed it. Those are the store's own tests.
    (out_dir / "context.json").write_text(json.dumps({
        "recent": [{"artist": play["artist"], "title": play["title"], "source": play["source"], "kind": play["kind"]}
                   for play in context.recent],
        "top_recent": [{"artist": artist, "count": count} for artist, count in context.top_recent],
        "top_all": [{"artist": artist, "count": count} for artist, count in context.top_all],
        "loved": context.loved,
        "rejected": context.rejected,
        "directions": context.directions,
        "kicked_artists": context.kicked_artists,
        "taste": context.taste,
    }, indent=2))
    variants = [
        ("plain", {"n": 6, "dig": 1}),
        ("deep", {"n": 6, "dig": 2}),
        ("follow", {"n": 4, "dig": 1, "direction_hint": "spiritual jazz, modal drift"}),
        ("far", {"n": 4, "dig": 1, "reach": "far"}),
        ("lean", {"n": 6, "dig": 1, "lean": "melancholic, Portuguese"}),
    ]
    for name, kwargs in variants:
        path = out_dir / f"prompt_{name}.txt"
        path.write_text(candidates_prompt(context, **kwargs))
        written.append({"file": path.name, "kwargs": kwargs})
    store.close()
    return written


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generated_from_commit": git_commit(),
        "sample_rate": SAMPLE_RATE,
        "tolerances": TOLERANCES,
        "front_end": {
            "n_fft": features.N_FFT, "hop": features.HOP, "n_mels": features.N_MELS,
            "clip_seconds": features.CLIP_S, "power_floor": features.POWER_FLOOR,
            "window": "hann-periodic", "padding": "reflect, n_fft // 2", "scale": "10 * log10",
            "mel_filters": "clap-mel.npy, shipped with the model",
        },
        "lcg": {"multiplier": LCG_MULTIPLIER, "increment": LCG_INCREMENT, "modulus": LCG_MODULUS},
        "waves": WAVE_SPECS,
        "logmel": [], "features": [], "embeddings": [], "prompts": [],
    }

    mel_filters = np.load(clap.DEFAULT_DIR / clap.MEL_FILE)
    manifest["mel_filters"] = write_f32(OUT_DIR / "mel_filters.f32", mel_filters)

    embedder = clap.Embedder() if clap.model_present() else None
    if embedder is None:
        print("! CLAP model not installed — writing everything except embeddings")

    waves = build_waves()
    sensitivity = max(ulp_sensitivity(wave.samples, mel_filters) for wave in waves)
    tolerance = max(TOLERANCES["logmel_max_abs_db"], 10.0 * sensitivity)
    manifest["tolerances"]["logmel_max_abs_db"] = tolerance
    manifest["tolerances"]["features_max_abs_db"] = tolerance
    manifest["one_ulp_sensitivity_db"] = sensitivity
    print(f"  log-mel tolerance {tolerance:.2e} dB = 10x the measured 1-ULP sensitivity {sensitivity:.2e} dB")

    for wave in waves:
        manifest["logmel"].append(write_f32(OUT_DIR / f"logmel_{wave.name}.f32",
                                            features.log_mel(wave.samples, mel_filters)))
        manifest["logmel"][-1]["wave"] = wave.name
        manifest["features"].append(write_f32(OUT_DIR / f"features_{wave.name}.f32",
                                              features.input_features(wave.samples, mel_filters)))
        manifest["features"][-1]["wave"] = wave.name
        if embedder is not None:
            manifest["embeddings"].append(write_f32(OUT_DIR / f"embedding_{wave.name}.f32",
                                                    embedder.embed_audio(wave.samples)))
            manifest["embeddings"][-1]["wave"] = wave.name
        print(f"  {wave.name}: {wave.description}")

    (OUT_DIR / "bands.json").write_text(json.dumps(bands_cases(), indent=2))
    manifest["bands"] = {"file": "bands.json"}
    manifest["prompts"] = prompt_fixtures(OUT_DIR)
    manifest["context"] = {"file": "context.json"}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_bytes = sum(path.stat().st_size for path in OUT_DIR.iterdir())
    print(f"wrote {len(list(OUT_DIR.iterdir()))} files, {total_bytes / 1e6:.1f} MB → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
