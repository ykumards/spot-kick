from collections.abc import Callable

import numpy as np
import pytest

from spotkick.brain import resolve
from spotkick.ears import clap, features, previews
from spotkick.memory.store import Store
from spotkick.player import spotify
from spotkick.player.spotify_api import SpotifyAPI, SpotifyAPIError

GOOD_URI = "spotify:track:3n3Ppam7vgaVa1iaRUc9Lp"
BAD_URI = "spotify:track:0000000000000000000000"
GOOD_ID = GOOD_URI.split(":")[-1]
PREVIEW_SAMPLES = 30 * features.SR


class FakeResponse:
    def __init__(self, status: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code: int = status
        self._payload: dict = payload or {}
        self.text: str = text

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


Handler = Callable[[str, dict | None], FakeResponse]


class FakeHTTP:
    """A ``previews.HttpClient`` whose every ``get`` is answered by ``handler`` and recorded in ``calls``."""

    def __init__(self, handler: Handler):
        self.handler = handler
        self.calls: list[tuple[str, dict | None]] = []

    def get(
        self, url: str, *, params: dict | None = None, headers: dict | None = None, timeout: float | None = None
    ) -> FakeResponse:
        self.calls.append((url, params))
        return self.handler(url, params)


def http_returning(results: list[dict]) -> FakeHTTP:
    return FakeHTTP(lambda url, params: FakeResponse(200, {"results": results}))


# ---------------------------------------------------------------- previews
def test_lookup_prefers_exact_artist_and_shortest_title():
    results = [
        {
            "artistName": "Azymuth",
            "trackName": "Linha do Horizonte (Live)",
            "trackId": 1,
            "previewUrl": "u1",
            "trackTimeMillis": 300000,
        },
        {
            "artistName": "Azymuth",
            "trackName": "Linha do Horizonte",
            "trackId": 2,
            "previewUrl": "u2",
            "trackTimeMillis": 270000,
            "collectionName": "Light as a Feather",
            "primaryGenreName": "Jazz",
        },
        {"artistName": "Some Tribute Band", "trackName": "Linha do Horizonte", "trackId": 3, "previewUrl": "u3"},
        {"artistName": "Azymuth", "trackName": "Linha do Horizonte", "trackId": 4, "previewUrl": None},
    ]
    found = previews.lookup("Azymuth", "Linha do Horizonte", session=http_returning(results))
    assert found is not None
    assert found.preview_url == "u2"
    assert found.duration_s == 270.0
    assert found.album == "Light as a Feather"


def test_lookup_rejects_unrelated_results():
    results = [
        {"artistName": "Someone Else", "trackName": "Linha do Horizonte", "trackId": 8, "previewUrl": "u"},
        {"artistName": "Azymuth", "trackName": "Different Song", "trackId": 9, "previewUrl": "u"},
    ]
    assert previews.lookup("Azymuth", "Linha do Horizonte", session=http_returning(results)) is None
    assert previews.lookup("X", "Y", session=http_returning([])) is None


# ---------------------------------------------------------------- clap cache
def test_embed_track_caches_in_store():
    store = Store(":memory:")
    track = store.upsert_track("A", "1", preview_url="http://x/p.m4a")
    embedded_urls = []

    class FakeEmbedder(clap.Embedder):
        def embed_url(self, preview_url: str) -> np.ndarray:
            embedded_urls.append(preview_url)
            return clap.normalize(np.arange(8, dtype=np.float32))

    embedder = FakeEmbedder()
    first = clap.embed_track(store, embedder, track)
    second = clap.embed_track(store, embedder, track)
    assert embedded_urls == ["http://x/p.m4a"]
    assert first is not None
    assert second is not None
    assert np.allclose(first, second)
    assert abs(np.linalg.norm(first) - 1) < 1e-6

    without_preview = store.upsert_track("B", "2")
    assert clap.embed_track(store, embedder, without_preview) is None


# ---------------------------------------------------------------- player
def test_to_uri_accepts_only_track_uris():
    assert spotify.to_uri(GOOD_URI) == GOOD_URI
    assert spotify.to_uri(f"spotify:album:{GOOD_ID}") is None
    with pytest.raises(ValueError):
        spotify.play("not a uri")


def test_parse_now_playing():
    line = "Zombie\tFela Kuti\tZombie\t732000\t61.5\tspotify:track:abcdefghijklmnopqrstuv\t63"
    track = spotify.parse_now_playing(line)
    assert track is not None
    assert track.label == "Fela Kuti — Zombie"
    assert track.duration_s == 732.0
    assert track.position_s == 61.5
    assert track.popularity == 63

    without_popularity = spotify.parse_now_playing("Zombie\tFela Kuti\tZombie\t732000\t61.5\tspotify:track:x")
    assert without_popularity is not None
    assert without_popularity.popularity is None
    assert spotify.parse_now_playing("garbage") is None
    with_art = spotify.parse_now_playing("Song\tArtist\tAlbum\t200000\t12.5\tspotify:track:x\t55\thttps://i.scdn.co/image/abc")
    assert with_art is not None and with_art.artwork_url == "https://i.scdn.co/image/abc"
    without_art = spotify.parse_now_playing("Song\tArtist\tAlbum\t200000\t12.5\tspotify:track:x\t55\tmissing value")
    assert without_art is not None and without_art.artwork_url is None


def test_spotify_is_never_addressed_without_the_running_guard(monkeypatch):
    """Every command runs under an ``is running`` check in the same script, so a closed Spotify raises PlayerError
    rather than being launched.
    """
    issued_scripts = []

    def spotify_is_closed(script: str) -> str:
        issued_scripts.append(script)
        return spotify.NOT_RUNNING_MARKER

    monkeypatch.setattr(spotify, "run_applescript", spotify_is_closed)
    with pytest.raises(spotify.PlayerError, match="isn't running"):
        spotify.now_playing()
    with pytest.raises(spotify.PlayerError):
        spotify.play(GOOD_URI)
    with pytest.raises(spotify.PlayerError):
        spotify.state()
    with pytest.raises(spotify.PlayerError):
        spotify.set_volume(30)
    assert len(issued_scripts) == 4
    for script in issued_scripts:
        assert script.startswith(f"if {spotify.IS_RUNNING_SCRIPT} then")
        assert "System Events" not in script


def test_play_and_confirm_reads_back(monkeypatch):
    issued_scripts = []

    def record_script(script: str) -> str:
        issued_scripts.append(script)
        return ""

    monkeypatch.setattr(spotify, "run_applescript", record_script)
    monkeypatch.setattr(spotify, "state", lambda: "playing")
    monkeypatch.setattr(spotify.time, "sleep", lambda seconds: None)

    monkeypatch.setattr(spotify, "now_playing", lambda: spotify.Track("x", "y", "z", 100, 1, GOOD_URI))
    assert spotify.play_and_confirm(GOOD_URI).uri == GOOD_URI
    assert f'play track "{GOOD_URI}"' in issued_scripts[0]

    monkeypatch.setattr(spotify, "now_playing", lambda: spotify.Track("x", "y", "z", 100, 1, "spotify:track:other"))
    with pytest.raises(spotify.PlayerError):
        spotify.play_and_confirm(GOOD_URI, timeout_s=0.01)


# ---------------------------------------------------------------- resolve


class FakeSpotifyHttp:
    """A fake token endpoint and ``/v1/search`` answering from a fixed hit list; records the requests."""

    def __init__(self, hits: list[dict], *, token_status: int = 200, search_status: int = 200):
        self.hits = hits
        self.token_status = token_status
        self.search_status = search_status
        self.token_requests = 0
        self.searches: list[dict] = []

    def post(self, url: str, *, data: dict | None = None, headers: dict | None = None, timeout: float | None = None):
        self.token_requests += 1
        return FakeResponse(self.token_status, {"access_token": "tok", "expires_in": 3600})

    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None, timeout: float | None = None):
        self.searches.append({"params": params, "headers": headers})
        return FakeResponse(self.search_status, {"tracks": {"items": self.hits}})


def hit(uri: str, artist: str, title: str) -> dict:
    return {"uri": uri, "name": title, "artists": [{"name": artist}]}


def test_names_match_ignores_accents_and_punctuation():
    assert resolve.names_match("Netsanet", "Nètsanèt")
    assert resolve.names_match("Aguia Nao Come Mosca", "Águia Não Come Mosca")
    assert resolve.names_match("Réu Confesso", "Reu Confesso - Remastered")
    assert not resolve.names_match("Netsanet", "Something Else")


def test_resolve_takes_the_first_hit_that_is_this_song():
    http = FakeSpotifyHttp([
        hit("spotify:track:" + "A" * 22, "Azymuth", "Linha do Horizonte - Live"),
        hit(GOOD_URI, "Azymuth", "Linha do Horizonte"),
    ])
    api = SpotifyAPI("id", "secret", session=http)
    found = resolve.resolve("Azymuth", "linha do horizonte", api)
    assert found is not None
    assert found.uri == "spotify:track:" + "A" * 22        # Spotify's order, filtered by name, not re-ranked
    assert http.searches[0]["params"]["q"] == "track:linha do horizonte artist:Azymuth"
    assert http.searches[0]["headers"]["Authorization"] == "Bearer tok"
    assert http.token_requests == 1
    resolve.resolve("Azymuth", "Linha do Horizonte", api)
    assert http.token_requests == 1                          # the token is cached

    other_song = SpotifyAPI("id", "secret", session=FakeSpotifyHttp([hit(GOOD_URI, "Someone Else", "Linha")]))
    assert resolve.resolve("Azymuth", "Linha do Horizonte", other_song) is None
    nothing = SpotifyAPI("id", "secret", session=FakeSpotifyHttp([]))
    assert resolve.resolve("Azymuth", "Linha do Horizonte", nothing) is None


def test_resolve_survives_spotify_being_unavailable():
    log_lines = []
    refused = SpotifyAPI("id", "wrong", session=FakeSpotifyHttp([], token_status=400))
    assert resolve.resolve("A", "B", refused, log=log_lines.append) is None
    assert "refused the credentials" in log_lines[0]
    with pytest.raises(SpotifyAPIError, match="credentials"):
        SpotifyAPI("", "", session=FakeSpotifyHttp([])).search_tracks("A", "B")
    assert not SpotifyAPI("", "", session=FakeSpotifyHttp([])).configured
    down = SpotifyAPI("id", "secret", session=FakeSpotifyHttp([], search_status=503))
    assert resolve.resolve("A", "B", down, log=log_lines.append) is None
    assert "HTTP 503" in log_lines[-1]


def noise_preview(rng: np.random.Generator) -> np.ndarray:
    return (0.2 * rng.standard_normal(PREVIEW_SAMPLES)).astype(np.float32)


@pytest.mark.skipif(not clap.model_present(), reason="clap-audio.onnx not installed")
def test_onnx_embedder_is_deterministic_and_unit():
    embedder = clap.Embedder()
    rng = np.random.default_rng(1)
    wave = noise_preview(rng)
    first = embedder.embed_audio(wave)
    second = embedder.embed_audio(wave)
    assert first.shape == (512,)
    assert abs(np.linalg.norm(first) - 1) < 1e-5
    assert np.allclose(first, second)

    short = embedder.embed_audio(wave[: 3 * features.SR])  # repeat-padded, single clip
    assert short.shape == (512,)

    other = embedder.embed_audio(noise_preview(rng))
    assert float(first @ other) < 0.999


def test_credentials_are_kept_in_the_keychain_not_the_config(monkeypatch, tmp_path):
    from spotkick import config
    from spotkick.player import keychain, spotify_api

    stored = {}
    commands = []

    def fake_security(*arguments):
        commands.append(arguments)
        if arguments[0] == "add-generic-password":
            stored["secret"] = arguments[arguments.index("-w") + 1]
            return FakeCompleted(0, "")
        if arguments[0] == "find-generic-password":
            if "secret" in stored:
                return FakeCompleted(0, stored["secret"] + "\n")
            return FakeCompleted(keychain.NOT_FOUND_EXIT_CODE, "")
        return FakeCompleted(0, "")

    monkeypatch.setattr(keychain, "run_security", fake_security)
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.delenv(spotify_api.SECRET_ENV_VAR, raising=False)

    assert keychain.get_secret() is None
    assert spotify_api.stored_secret() == ""
    refused = FakeSpotifyHttp([], token_status=401)
    with pytest.raises(SpotifyAPIError):
        spotify_api.save_credentials("id", "bad", session=refused)
    assert "secret" not in stored                                      # nothing is stored when Spotify refuses

    spotify_api.save_credentials("id", "s3cret", session=FakeSpotifyHttp([]))
    assert stored["secret"] == "s3cret"
    assert "s3cret" not in (tmp_path / "config.toml").read_text()      # the file has the id only
    assert 'spotify_client_id = "id"' in (tmp_path / "config.toml").read_text()
    assert spotify_api.stored_secret() == "s3cret"

    monkeypatch.setenv(spotify_api.SECRET_ENV_VAR, "from-env")
    assert spotify_api.stored_secret() == "from-env"                   # the environment wins for terminal runs


def test_set_credentials_forgets_the_old_token():
    http = FakeSpotifyHttp([hit(GOOD_URI, "Azymuth", "Linha do Horizonte")])
    api = SpotifyAPI("id", "secret", session=http)
    api.search_tracks("Azymuth", "Linha do Horizonte")
    assert http.token_requests == 1
    api.set_credentials("id2", "secret2")
    api.search_tracks("Azymuth", "Linha do Horizonte")
    assert http.token_requests == 2


class FakeCompleted:
    def __init__(self, returncode: int, stdout: str, stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_wav_samples_reads_channels_and_data():
    import struct

    from spotkick.ears.clap import wav_samples

    frames = struct.pack("<6f", 1.0, 0.0, 0.5, 0.5, -1.0, 1.0)      # 3 stereo frames
    fmt = struct.pack("<HHIIHH", 3, 2, 48000, 48000 * 8, 8, 32)
    wav = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(frames)) + b"WAVE"
    wav += b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(frames)) + frames
    channels, samples = wav_samples(wav)
    assert channels == 2
    assert samples.tolist() == [1.0, 0.0, 0.5, 0.5, -1.0, 1.0]
