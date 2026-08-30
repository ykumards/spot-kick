import numpy as np

from spotkick.ears import features

PREVIEW_SAMPLES = 30 * features.SR


def diagonal_mel_filters() -> np.ndarray:
    """A (513, 64) filterbank where each mel bin selects one FFT bin, so silence gives exactly the floor."""
    mel_filters = np.zeros((513, features.N_MELS), dtype=np.float32)
    mel_filters[np.arange(features.N_MELS) * 8, np.arange(features.N_MELS)] = 1.0
    return mel_filters


def test_log_mel_shape_and_floor():
    silence = np.zeros(features.CLIP, dtype=np.float32)
    log_mel = features.log_mel(silence, diagonal_mel_filters())
    assert log_mel.shape == (1001, features.N_MELS)
    assert np.all(log_mel == -100.0)


def test_clips_are_deterministic_and_cover_the_preview():
    ramp = np.arange(PREVIEW_SAMPLES, dtype=np.float32)
    three = features.clips(ramp, 3)
    first_samples = [clip[0] for clip in three]
    assert first_samples == [0, (PREVIEW_SAMPLES - features.CLIP) // 2, PREVIEW_SAMPLES - features.CLIP]
    assert all(len(clip) == features.CLIP for clip in three)

    exact_length = features.clips(ramp[: features.CLIP], 3)
    assert len(exact_length) == 1

    four_seconds = 4 * features.SR
    short = features.clips(ramp[:four_seconds], 3)
    assert len(short) == 1
    assert len(short[0]) == features.CLIP
    assert short[0][four_seconds] == 0.0  # repeat-padded: the ramp starts over
