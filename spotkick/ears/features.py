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
POWER_FLOOR = 1e-10
DEFAULT_N_CLIPS = 3


def hann_periodic(length: int) -> np.ndarray:
    """The periodic Hann window transformers uses (denominator `length`, not `length - 1`)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)


def log_mel(wave: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """(samples,) float32 → (frames, 64) float32 log-mel, transformers conventions."""
    samples = np.asarray(wave, dtype=np.float64)
    samples = np.pad(samples, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(samples) - N_FFT) // HOP
    frame_indices = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = samples[frame_indices] * hann_periodic(N_FFT)[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = power @ mel_filters
    return (10.0 * np.log10(np.maximum(mel, POWER_FLOOR))).astype(np.float32)


def clips(wave: np.ndarray, n: int = DEFAULT_N_CLIPS) -> list[np.ndarray]:
    """Deterministic 10 s crops: start / middle / end of the preview (the reference extractor takes a *random* crop,
    which made the old embeddings nondeterministic). Short audio is repeat-padded like the reference does."""
    samples = np.asarray(wave, dtype=np.float32)
    if len(samples) < CLIP:
        repeats = int(np.ceil(CLIP / max(len(samples), 1)))
        padded = np.tile(samples, repeats)[:CLIP]
        return [padded]
    if len(samples) == CLIP or n == 1:
        middle_start = (len(samples) - CLIP) // 2
        return [samples[middle_start:][:CLIP]]
    starts = np.linspace(0, len(samples) - CLIP, n).astype(int)
    return [samples[start:start + CLIP] for start in starts]


def input_features(wave: np.ndarray, mel_filters: np.ndarray, n_clips: int = DEFAULT_N_CLIPS) -> np.ndarray:
    """(n_clips, 1, 1001, 64) ready for the ONNX audio tower."""
    per_clip = [log_mel(clip, mel_filters)[None] for clip in clips(wave, n_clips)]
    return np.stack(per_clip).astype(np.float32)
