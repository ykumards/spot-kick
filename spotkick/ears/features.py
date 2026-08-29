"""CLAP's log-mel front end in plain numpy, so the runtime needs onnxruntime and nothing else.

Matches transformers' `ClapFeatureExtractor` for the unfused model: Hann (periodic) window of 1024, hop 480,
centered reflect padding, power spectrum, Slaney mel filterbank (64 bins, 50–14000 Hz, shipped next to the
ONNX file), 10·log10 with a 1e-10 floor. Ten seconds of 48 kHz audio → (1001, 64).
"""
from __future__ import annotations

import numpy as np

SR = 48000
N_FFT = 1024
HOP = 480
N_MELS = 64
CLIP_S = 10
CLIP = SR * CLIP_S


def hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def log_mel(wave: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """(samples,) float32 → (frames, 64) float32 log-mel, transformers conventions."""
    x = np.asarray(wave, dtype=np.float64)
    x = np.pad(x, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * hann_periodic(N_FFT)[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2          # (frames, 513)
    mel = power @ mel_filters                                # (frames, 64)
    return (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)


def clips(wave: np.ndarray, n: int = 3) -> list[np.ndarray]:
    """Deterministic 10 s crops: start / middle / end of the preview (the reference extractor takes a *random* crop,
    which made the old embeddings nondeterministic). Short audio is repeat-padded like the reference does."""
    w = np.asarray(wave, dtype=np.float32)
    if len(w) < CLIP:
        reps = int(np.ceil(CLIP / max(len(w), 1)))
        w = np.tile(w, reps)[:CLIP]
        return [w]
    if len(w) == CLIP or n == 1:
        return [w[(len(w) - CLIP) // 2:][:CLIP]]
    starts = np.linspace(0, len(w) - CLIP, n).astype(int)
    return [w[s:s + CLIP] for s in starts]


def input_features(wave: np.ndarray, mel_filters: np.ndarray, n_clips: int = 3) -> np.ndarray:
    """(n_clips, 1, 1001, 64) ready for the ONNX audio tower."""
    return np.stack([log_mel(c, mel_filters)[None] for c in clips(wave, n_clips)]).astype(np.float32)
