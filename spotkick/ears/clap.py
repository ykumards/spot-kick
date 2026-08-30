"""The ruler: CLAP audio embeddings of 30-second previews, run with onnxruntime. Every distance in Spot Kick
is measured here. The LLM never sees these vectors and cannot grade its own work.

Runtime needs: `clap-audio.onnx` (the HTSAT audio tower + projection, fp16, 59 MB, exported by
`scripts/export_clap_onnx.py`) and `clap-mel.npy` (its mel filterbank) in `~/.spotkick/models/`. Previews are
decoded with macOS's own `afconvert`, so there is nothing to brew. No torch, no transformers. Vectors are cached
in the store, so a track is embedded once.
"""
from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import HOME
from . import features

if TYPE_CHECKING:  # onnxruntime and the store are runtime imports deferred below; only their types are needed here
    import onnxruntime

    from ..mind.store import Store, Track

MODEL_TAG = "clap-htsat-unfused-onnx"
ONNX_FILE = "clap-audio.onnx"
MEL_FILE = "clap-mel.npy"
MODEL_FILES = (ONNX_FILE, MEL_FILE)
# The `models-v1` release of the repo; downloadable without auth once the repo is public.
MODEL_URL = "https://github.com/ykumards/spot-kick/releases/download/models-v1/"
DEFAULT_DIR = HOME / "models"
DEFAULT_PROVIDERS = ["CPUExecutionProvider"]
DOWNLOAD_CHUNK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT_S = 60
PREVIEW_TIMEOUT_S = 30
ONNX_INPUT_NAME = "input_features"


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length; the zero vector is returned untouched."""
    length = np.linalg.norm(vector)
    if length > 0:
        return vector / length
    return vector


def model_present(model_dir: Path = DEFAULT_DIR) -> bool:
    return all((model_dir / filename).exists() for filename in MODEL_FILES)


def download_file(url: str, destination: Path) -> None:
    """Stream `url` into `destination` via a `.part` file, so a killed download never leaves a truncated model."""
    import requests  # deferred: only needed on first run, keeps app startup fast

    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
        partial.rename(destination)


def ensure_model(model_dir: Path = DEFAULT_DIR, log: Callable[[str], None] = lambda message: None) -> Path:
    """Download whichever model files are missing from `model_dir`. Returns the directory."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        destination = model_dir / filename
        if destination.exists():
            continue
        log(f"downloading {filename}…")
        download_file(MODEL_URL + filename, destination)
    return model_dir


class Embedder:
    """Lazily loads the ONNX audio tower; nothing heavy happens until the first `embed_audio`."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_DIR,
        *,
        n_clips: int = features.DEFAULT_N_CLIPS,
        providers: list[str] | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.n_clips = n_clips
        self.providers = providers or DEFAULT_PROVIDERS
        self._session: onnxruntime.InferenceSession | None = None
        self._mel_filters: np.ndarray | None = None

    def _load(self) -> tuple[onnxruntime.InferenceSession, np.ndarray]:
        if self._session is not None and self._mel_filters is not None:
            return self._session, self._mel_filters
        import onnxruntime  # deferred: importing onnxruntime costs real startup time

        ensure_model(self.model_dir)
        mel_filters = np.load(self.model_dir / MEL_FILE)
        session = onnxruntime.InferenceSession(str(self.model_dir / ONNX_FILE), providers=self.providers)
        self._mel_filters = mel_filters
        self._session = session
        return session, mel_filters

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def embed_audio(self, wave: np.ndarray) -> np.ndarray:
        """Mono float32 at 48 kHz → unit vector (512). Mean of up to `n_clips` deterministic 10 s crops."""
        session, mel_filters = self._load()
        model_input = features.input_features(wave, mel_filters, self.n_clips)
        per_clip = np.asarray(session.run(None, {ONNX_INPUT_NAME: model_input})[0], dtype=np.float32)
        per_clip = per_clip / np.linalg.norm(per_clip, axis=1, keepdims=True)
        return normalize(per_clip.mean(axis=0).astype(np.float32))

    def embed_url(self, preview_url: str) -> np.ndarray:
        import requests  # deferred: keeps the module importable without network deps at startup

        raw = requests.get(preview_url, timeout=PREVIEW_TIMEOUT_S).content
        return self.embed_audio(decode(raw))


WAV_HEADER_BYTES = 44
AFCONVERT_TIMEOUT_S = 60


def decode(raw: bytes) -> np.ndarray:
    """m4a bytes → mono float32 at 48 kHz, via macOS's `afconvert` (ships with the system).

    Channels are mixed here, not by afconvert: its `-c 1` sums them, ffmpeg's `-ac 1` (which every stored embedding
    and the reference implementation used) mixes stereo as (L + R) / √2, and CLAP is not level-invariant, so the
    same convention is kept to the bit."""
    with tempfile.TemporaryDirectory() as scratch_dir:
        source = Path(scratch_dir) / "preview.m4a"
        target = Path(scratch_dir) / "preview.wav"
        source.write_bytes(raw)
        command = ["afconvert", "-f", "WAVE", "-d", f"LEF32@{features.SR}", str(source), str(target)]
        subprocess.run(command, capture_output=True, check=True, timeout=AFCONVERT_TIMEOUT_S)
        channels, samples = wav_samples(target.read_bytes())
    if channels == 1:
        return samples
    return (samples.reshape(-1, channels).sum(axis=1) / np.sqrt(channels)).astype(np.float32)


def wav_samples(wav: bytes) -> tuple[int, np.ndarray]:
    """(channel count, interleaved float32 samples) of a WAVE file, found by walking the RIFF chunks."""
    channels = 1
    offset = 12
    while offset + 8 <= len(wav):
        chunk_id = wav[offset:offset + 4]
        chunk_size = int.from_bytes(wav[offset + 4:offset + 8], "little")
        if chunk_id == b"fmt ":
            channels = int.from_bytes(wav[offset + 10:offset + 12], "little")
        elif chunk_id == b"data":
            return channels, np.frombuffer(wav[offset + 8:offset + 8 + chunk_size], dtype="<f4").copy()
        offset += 8 + chunk_size + (chunk_size % 2)
    raise ValueError("no data chunk in the decoded audio")


def embed_track(store: Store, embedder: Embedder, track: Track) -> np.ndarray | None:
    """Vector for a store Track: cached in `embeddings`, else download the preview, embed, cache. None if no preview."""
    cached = store.embedding(track.id)
    if cached is not None:
        return cached
    if not track.preview_url:
        return None
    vector = embedder.embed_url(track.preview_url)
    store.put_embedding(track.id, vector, MODEL_TAG)
    return vector
