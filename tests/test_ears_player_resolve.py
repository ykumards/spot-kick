import numpy as np
import pytest

from spotkick.brain import resolve as R
from spotkick.ears import clap, previews
from spotkick.mind.store import Store
from spotkick.player import spotify as P


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload = status, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeHTTP:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.handler(url, params)


# ---------------------------------------------------------------- previews
def test_lookup_prefers_exact_artist_and_shortest_title():
    results = [
        {"artistName": "Azymuth", "trackName": "Linha do Horizonte (Live)", "trackId": 1, "previewUrl": "u1", "trackTimeMillis": 300000},
        {"artistName": "Azymuth", "trackName": "Linha do Horizonte", "trackId": 2, "previewUrl": "u2", "trackTimeMillis": 270000,
         "collectionName": "Light as a Feather", "primaryGenreName": "Jazz"},
        {"artistName": "Some Tribute Band", "trackName": "Linha do Horizonte", "trackId": 3, "previewUrl": "u3"},
        {"artistName": "Azymuth", "trackName": "Linha do Horizonte", "trackId": 4, "previewUrl": None},
    ]
    http = FakeHTTP(lambda u, p: FakeResp(200, {"results": results}))
    got = previews.lookup("Azymuth", "Linha do Horizonte", session=http)
    assert got.itunes_id == 2 and got.preview_url == "u2" and got.duration_s == 270.0 and got.album == "Light as a Feather"


def test_lookup_rejects_unrelated_results():
    results = [{"artistName": "Someone Else", "trackName": "Different Song", "trackId": 9, "previewUrl": "u"}]
    http = FakeHTTP(lambda u, p: FakeResp(200, {"results": results}))
    assert previews.lookup("Azymuth", "Linha do Horizonte", session=http) is None
    assert previews.lookup("X", "Y", session=FakeHTTP(lambda u, p: FakeResp(200, {"results": []}))) is None


# ---------------------------------------------------------------- clap cache
def test_embed_track_caches_in_store(monkeypatch):
    s = Store(":memory:")
    t = s.upsert_track("A", "1", preview_url="http://x/p.m4a")
    calls = []

    class E(clap.Embedder):
        def embed_url(self, url):
            calls.append(url)
            return clap.normalize(np.arange(8, dtype=np.float32))

    e = E()
    v1 = clap.embed_track(s, e, t)
    v2 = clap.embed_track(s, e, t)
    assert calls == ["http://x/p.m4a"] and np.allclose(v1, v2) and abs(np.linalg.norm(v1) - 1) < 1e-6
    assert clap.embed_track(s, e, s.upsert_track("B", "2")) is None  # no preview
    assert not e.loaded


# ---------------------------------------------------------------- player
def test_to_uri_accepts_links_and_uris():
    assert P.to_uri("spotify:track:3n3Ppam7vgaVa1iaRUc9Lp") == "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"
    assert P.to_uri("https://open.spotify.com/intl-de/track/3n3Ppam7vgaVa1iaRUc9Lp?si=abc") == "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"
    assert P.to_uri("https://open.spotify.com/album/3n3Ppam7vgaVa1iaRUc9Lp") is None
    with pytest.raises(ValueError):
        P.play("not a uri")


def test_parse_now_playing():
    t = P.parse_now_playing("Zombie\tFela Kuti\tZombie\t732000\t61.5\tspotify:track:abcdefghijklmnopqrstuv\t63")
    assert t.label == "Fela Kuti — Zombie" and t.duration_s == 732.0 and t.position_s == 61.5 and t.popularity == 63
    assert t.remaining_s == 670.5
    assert P.parse_now_playing("Zombie\tFela Kuti\tZombie\t732000\t61.5\tspotify:track:x").popularity is None
    assert P.parse_now_playing("garbage") is None


def test_play_and_confirm_reads_back(monkeypatch):
    issued = []
    monkeypatch.setattr(P, "_osa", lambda script: issued.append(script) or "")
    monkeypatch.setattr(P, "state", lambda: "playing")
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    uri = "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"
    monkeypatch.setattr(P, "now_playing", lambda: P.Track("x", "y", "z", 100, 1, uri))
    assert P.play_and_confirm(uri).uri == uri and 'play track "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"' in issued[0]
    monkeypatch.setattr(P, "now_playing", lambda: P.Track("x", "y", "z", 100, 1, "spotify:track:other"))
    with pytest.raises(P.PlayerError):
        P.play_and_confirm(uri, timeout_s=0.01)


# ---------------------------------------------------------------- resolve
GOOD = "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"
BAD = "spotify:track:0000000000000000000000"


def oembed(url, params):
    if GOOD.split(":")[-1] in params["url"]:
        return FakeResp(200, {"title": "Linha do Horizonte"})
    return FakeResp(404)


def test_validate_via_oembed():
    http = FakeHTTP(oembed)
    assert R.validate(GOOD, "Linha do Horizonte", session=http)
    assert R.validate(GOOD, "linha do horizonte (remastered)", session=http)
    assert not R.validate(GOOD, "Completely Different", session=http)
    assert not R.validate(BAD, "Linha do Horizonte", session=http)
    assert not R.validate("nonsense", "Linha do Horizonte", session=http)
    assert not R.validate(None, "Linha do Horizonte", session=http)


def test_resolve_memory_then_search():
    http = FakeHTTP(oembed)
    r = R.resolve("Azymuth", "Linha do Horizonte", GOOD, session=http)
    assert r.how == "memory" and r.uri == GOOD
    searched = []
    r = R.resolve("Azymuth", "Linha do Horizonte", BAD, searcher=lambda a, t: searched.append((a, t)) or GOOD, session=http)
    assert r.how == "search" and searched == [("Azymuth", "Linha do Horizonte")]
    assert R.resolve("Azymuth", "Linha do Horizonte", BAD, searcher=lambda a, t: BAD, session=http) is None
    assert R.resolve("Azymuth", "Linha do Horizonte", BAD, session=http) is None


# ---------------------------------------------------------------- onnx runtime (only when the model is installed)
@pytest.mark.skipif(not clap.model_present(), reason="clap-audio.onnx not installed")
def test_onnx_embedder_is_deterministic_and_unit():
    e = clap.Embedder()
    rng = np.random.default_rng(1)
    wave = (0.2 * rng.standard_normal(30 * clap.F.SR)).astype(np.float32)
    a, b = e.embed_audio(wave), e.embed_audio(wave)
    assert a.shape == (512,) and abs(np.linalg.norm(a) - 1) < 1e-5 and np.allclose(a, b)
    short = e.embed_audio(wave[: 3 * clap.F.SR])          # repeat-padded, single clip
    assert short.shape == (512,)
    other = e.embed_audio((0.2 * rng.standard_normal(30 * clap.F.SR)).astype(np.float32))
    assert float(a @ other) < 0.999
