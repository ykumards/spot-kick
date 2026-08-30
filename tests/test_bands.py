import itertools

import numpy as np

from spotkick.kick import bands

DIM = 16


def unit(seed: int) -> np.ndarray:
    return bands.normalize(np.random.default_rng(seed).standard_normal(DIM).astype(np.float32))


def test_strength_target_are_monotonic():
    assert [bands.strength_for(magnitude) for magnitude in (0.1, 0.5, 0.9)] == ["tap", "kick", "boot"]
    targets = [bands.target_for(magnitude) for magnitude in np.linspace(0, 1, 21)]
    assert all(later >= earlier for earlier, later in itertools.pairwise(targets))
    assert abs(bands.target_for(0.5) - 0.75) < 1e-9


def test_state_update_and_scale():
    state = bands.ListenerState(alpha=0.7)
    assert state.scale() == bands.DEFAULT_SCALE
    assert state.distance(unit(1)) == 0.0
    state.update(unit(1))
    assert state.vector is not None
    assert np.allclose(state.vector, unit(1))
    before = state.vector.copy()
    state.update(unit(2))
    expected = bands.normalize(0.7 * before + 0.3 * unit(2))
    assert np.allclose(state.vector, expected)
    state.update(unit(3))
    step, far = state.scale()
    assert 0 < step <= far
    assert state.band_for(step) == "tap"
    assert state.band_for(far * 1.5) == "boot"
    assert abs(state.rel(step)) < 1e-9
    assert abs(state.rel(far) - 1) < 1e-9


def test_measure_and_choose_by_target():
    state = bands.ListenerState()
    for seed in range(6):
        state.update(unit(seed))
    assert state.vector is not None
    candidates = [unit(100), unit(101), state.vector.copy(), unit(102)]
    measured = bands.measure(state, candidates)
    assert [candidate.index for candidate in measured] == [0, 1, 2, 3]
    assert measured[2].distance < 1e-6
    assert measured[2].rel < 0
    nearest = bands.choose(measured, target_rel=measured[2].rel)
    assert nearest is not None
    assert nearest.index == 2                            # the state itself is the nearest thing to "no kick"
    farthest = bands.choose(measured, target_rel=10.0)
    assert farthest is not None
    assert farthest.index == int(np.argmax([candidate.rel for candidate in measured]))
    assert bands.choose([], 0.5) is None


def test_followed_projection():
    pre = unit(1)
    kick = unit(2)
    assert abs(bands.followed(pre, kick, pre)) < 1e-6
    assert abs(bands.followed(pre, kick, kick) - 1.0) < 1e-6
    midway = pre + 0.4 * (kick - pre)
    assert abs(bands.followed(pre, kick, midway) - 0.4) < 1e-6
    recoiled = pre - 0.2 * (kick - pre)
    assert bands.followed(pre, kick, recoiled) < 0
    assert bands.followed(pre, pre, kick) == 0.0         # degenerate kick
    assert bands.verdict(0.9, 1) == "listening"
    assert [bands.verdict(fraction, 2) for fraction in (0.1, 0.4, 0.8)] == ["returned", "bent", "followed"]
