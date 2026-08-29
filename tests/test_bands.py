import itertools

import numpy as np

from spotkick.brain.spotify_api import SpotifySearch
from spotkick.kick import bands as B


def unit(seed, dim=16):
    return B.normalize(np.random.default_rng(seed).standard_normal(dim).astype(np.float32))


def test_strength_target_dose_are_monotonic():
    ms = np.linspace(0, 1, 21)
    assert [B.strength_for(m) for m in (0.1, 0.5, 0.9)] == ["tap", "kick", "boot"]
    ts = [B.target_for(m) for m in ms]
    assert all(b >= a for a, b in itertools.pairwise(ts)) and abs(B.target_for(0.5) - 0.75) < 1e-9
    assert [B.dose_for(m) for m in (0.1, 0.5, 0.9)] == [1, 3, 5]
    assert B.acceptance(0) == 0.75 and B.acceptance(1) == 0.63 and B.acceptance(5) == 0.63


def test_state_update_and_scale():
    st = B.ListenerState(alpha=0.7)
    assert st.scale() == B.DEFAULT_SCALE and st.distance(unit(1)) == 0.0
    st.update(unit(1))
    assert np.allclose(st.vector, unit(1))
    v0 = st.vector.copy(); st.update(unit(2))
    expected = B.normalize(0.7 * v0 + 0.3 * unit(2))
    assert np.allclose(st.vector, expected)
    st.update(unit(3), weight=0.3)                      # a skip pulls less
    step, far = st.scale()
    assert 0 < step <= far and st.band_for(step) == "tap" and st.band_for(far * 1.5) == "boot"
    assert abs(st.rel(step)) < 1e-9 and abs(st.rel(far) - 1) < 1e-9


def test_measure_and_choose_by_target():
    st = B.ListenerState()
    for i in range(6):
        st.update(unit(i))
    cands = [unit(100), unit(101), st.vector.copy(), unit(102)]
    m = B.measure(st, cands)
    assert [x.index for x in m] == [0, 1, 2, 3] and m[2].distance < 1e-6 and m[2].rel < 0
    near = B.choose(m, target_rel=m[2].rel)
    assert near.index == 2                               # the state itself is the nearest thing to "no kick"
    far = B.choose(m, target_rel=10.0)
    assert far.index == int(np.argmax([x.rel for x in m]))
    assert B.choose([], 0.5) is None


def test_followed_projection():
    pre, kick = unit(1), unit(2)
    assert abs(B.followed(pre, kick, pre)) < 1e-6
    assert abs(B.followed(pre, kick, kick) - 1.0) < 1e-6
    mid = pre + 0.4 * (kick - pre)
    assert abs(B.followed(pre, kick, mid) - 0.4) < 1e-6
    assert B.followed(pre, kick, pre - 0.2 * (kick - pre)) < 0
    assert B.followed(pre, pre, kick) == 0.0             # degenerate kick
    assert B.verdict(0.9, 1) == "listening"
    assert [B.verdict(f, 2) for f in (0.1, 0.4, 0.8)] == ["returned", "bent", "followed"]


def test_spotify_search_scores_and_needs_credentials():
    calls = []

    class Sess:
        def post(self, url, data=None, headers=None, timeout=None):
            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"access_token": "tok", "expires_in": 3600}
            return R()

        def get(self, url, params=None, headers=None, timeout=None):
            calls.append((params, headers))

            class R:
                status_code = 200

                def json(self):
                    return {"tracks": {"items": [
                        {"name": "Linha do Horizonte", "uri": "spotify:track:cover", "artists": [{"name": "Some Cover Band"}], "popularity": 90},
                        {"name": "Linha do Horizonte - Live", "uri": "spotify:track:live", "artists": [{"name": "Azymuth"}], "popularity": 20},
                        {"name": "Linha do Horizonte", "uri": "spotify:track:studio", "artists": [{"name": "Azymuth"}], "popularity": 55},
                    ]}}
            return R()

    assert SpotifySearch(client_id=None, client_secret=None, session=Sess())("Azymuth", "Linha do Horizonte") is None
    s = SpotifySearch("id", "secret", session=Sess())
    assert s("Azymuth", "Linha do Horizonte") == "spotify:track:studio"
    assert calls[0][1]["Authorization"] == "Bearer tok" and "artist:Azymuth" in calls[0][0]["q"]
    s("Azymuth", "Linha do Horizonte")
    assert len(calls) == 2                               # token reused
