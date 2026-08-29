"""The ruler: CLAP audio embeddings of 30-second previews, run with onnxruntime. Every distance in Spot Kick
is measured here. The LLM never sees these vectors and cannot grade its own work.

Runtime needs: `clap-audio.onnx` (the HTSAT audio tower + projection, 116 MB, exported by
`scripts/export_clap_onnx.py`) and `clap-mel.npy` (its mel filterbank) in `~/.spotkick/models/`, plus ffmpeg
for decoding. No torch, no transformers. Vectors are cached in the store, so a track is embedded once.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import features as F

MODEL_TAG = "clap-htsat-unfused-onnx"
MODEL_FILES = ("clap-audio.onnx", "clap-mel.npy")
# TODO: point at the GitHub release asset once the repo is public.
MODEL_URL = "https://github.com/ykumards/spot-kick/releases/download/models-v1/"
DEFAULT_DIR = Path.home() / ".spotkick" / "models"


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def model_present(model_dir: Path = DEFAULT_DIR) -> bool:
    return all((model_dir / f).exists() for f in MODEL_FILES)


def ensure_model(model_dir: Path = DEFAULT_DIR, log=lambda m: None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    for f in MODEL_FILES:
        p = model_dir / f
        if p.exists():
            continue
        import requests
        log(f"downloading {f}…")
        with requests.get(MODEL_URL + f, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = p.with_suffix(p.suffix + ".part")
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.rename(p)
    return model_dir


class Embedder:
    def __init__(self, model_dir: Path = DEFAULT_DIR, *, n_clips: int = 3, providers: list[str] | None = None):
        self.model_dir, self.n_clips = Path(model_dir), n_clips
        self.providers = providers or ["CPUExecutionProvider"]
        self._sess = self._mel = None

    def _load(self):
        if self._sess is None:
            import onnxruntime as ort
            ensure_model(self.model_dir)
            self._mel = np.load(self.model_dir / "clap-mel.npy")
            self._sess = ort.InferenceSession(str(self.model_dir / "clap-audio.onnx"), providers=self.providers)
        return self._sess, self._mel

    @property
    def loaded(self) -> bool:
        return self._sess is not None

    def embed_audio(self, wave: np.ndarray) -> np.ndarray:
        """Mono float32 at 48 kHz → unit vector (512). Mean of up to `n_clips` deterministic 10 s crops."""
        sess, mel = self._load()
        x = F.input_features(wave, mel, self.n_clips)
        out = sess.run(None, {"input_features": x})[0]                    # (clips, 512), un-normalized
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return normalize(out.mean(axis=0).astype(np.float32))

    def embed_url(self, preview_url: str) -> np.ndarray:
        import requests
        raw = requests.get(preview_url, timeout=30).content
        return self.embed_audio(decode(raw))


def decode(raw: bytes) -> np.ndarray:
    """m4a bytes → mono float32 at 48 kHz, via ffmpeg (Homebrew)."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "preview.m4a"
        src.write_bytes(raw)
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(F.SR), "-f", "f32le", "-"],
                             capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def embed_track(store, embedder: Embedder, track) -> np.ndarray | None:
    """Vector for a store Track: cached in `embeddings`, else download the preview, embed, cache. None if no preview."""
    v = store.embedding(track.id)
    if v is not None:
        return v
    if not track.preview_url:
        return None
    v = embedder.embed_url(track.preview_url)
    store.put_embedding(track.id, v, MODEL_TAG)
    return v
