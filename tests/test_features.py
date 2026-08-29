import numpy as np

from spotkick.ears import features as F


def test_log_mel_shape_and_floor():
    mel = np.zeros((513, 64), dtype=np.float32); mel[np.arange(64) * 8, np.arange(64)] = 1.0
    out = F.log_mel(np.zeros(F.CLIP, dtype=np.float32), mel)
    assert out.shape == (1001, 64) and np.all(out == -100.0)


def test_clips_are_deterministic_and_cover_the_preview():
    w = np.arange(30 * F.SR, dtype=np.float32)
    c = F.clips(w, 3)
    assert [x[0] for x in c] == [0, (30 * F.SR - F.CLIP) // 2, 30 * F.SR - F.CLIP] and all(len(x) == F.CLIP for x in c)
    assert len(F.clips(w[: F.CLIP], 3)) == 1
    s = F.clips(w[: 4 * F.SR], 3)
    assert len(s) == 1 and len(s[0]) == F.CLIP and s[0][4 * F.SR] == 0.0  # repeat-padded
