"""Spotify's Web API, with the developer's own credentials: the only way a name becomes a track id.

Client Credentials flow: the client id (config.toml) and secret (the Keychain, or the environment for terminal
runs) are exchanged over TLS for an app token — no user login, no browser — which is cached until it expires.
Only `/v1/search` is used. Each developer registers their own app at developer.spotify.com; nothing is shipped in
the repo, and the secret is never written to a file of ours.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Protocol

import requests

from ..config import Config
from . import keychain

SECRET_ENV_VAR = "SPOTKICK_SPOTIFY_CLIENT_SECRET"

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"
TIMEOUT_S = 15
SEARCH_LIMIT = 5
TOKEN_SAFETY_MARGIN_S = 60
NOT_CONFIGURED_MESSAGE = "no Spotify credentials: add your app's client id and secret in settings (see README)"


class HttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    def json(self) -> dict: ...


class HttpClient(Protocol):
    """`requests`, a `requests.Session`, or a test fake."""

    def get(
        self, url: str, *, params: dict | None = ..., headers: dict | None = ..., timeout: float | None = ...
    ) -> HttpResponse: ...

    def post(
        self, url: str, *, data: dict | None = ..., headers: dict | None = ..., timeout: float | None = ...
    ) -> HttpResponse: ...


class SpotifyAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class FoundTrack:
    uri: str
    artist: str
    title: str


class SpotifyAPI:
    def __init__(self, client_id: str, client_secret: str, *, session: HttpClient | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.http: HttpClient = session or requests.Session()
        self._token: str | None = None
        self._token_expires_at = 0.0

    @classmethod
    def from_config(cls, cfg: Config, *, session: HttpClient | None = None) -> SpotifyAPI:
        return cls(cfg.spotify_client_id, stored_secret(), session=session)

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def set_credentials(self, client_id: str, client_secret: str) -> None:
        """Swap credentials in place (the session keeps its reference); the next call fetches a fresh token."""
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._token_expires_at = 0.0

    def token(self) -> str:
        """The cached app token, refreshed a minute before Spotify says it expires."""
        if not self.configured:
            raise SpotifyAPIError(NOT_CONFIGURED_MESSAGE)
        if self._token is not None and time.time() < self._token_expires_at:
            return self._token
        credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}
        response = self.http.post(TOKEN_URL, data={"grant_type": "client_credentials"}, headers=headers,
                                  timeout=TIMEOUT_S)
        if response.status_code != 200:
            raise SpotifyAPIError(f"Spotify refused the credentials (HTTP {response.status_code}); check config.toml")
        payload = response.json()
        token = str(payload["access_token"])
        self._token = token
        self._token_expires_at = time.time() + float(payload.get("expires_in", 3600)) - TOKEN_SAFETY_MARGIN_S
        return token

    def search_tracks(self, artist: str, title: str) -> list[FoundTrack]:
        """Spotify's best matches for the name, in its order. Raises when Spotify cannot be asked."""
        params = {"q": f"track:{title} artist:{artist}", "type": "track", "limit": SEARCH_LIMIT}
        headers = {"Authorization": f"Bearer {self.token()}"}
        try:
            response = self.http.get(SEARCH_URL, params=params, headers=headers, timeout=TIMEOUT_S)
        except requests.RequestException as error:
            raise SpotifyAPIError(f"Spotify unreachable: {error}") from error
        if response.status_code != 200:
            raise SpotifyAPIError(f"Spotify search answered HTTP {response.status_code}")
        items = response.json().get("tracks", {}).get("items", [])
        return [found_track(item) for item in items if item.get("uri")]


def found_track(item: dict) -> FoundTrack:
    artists = ", ".join(performer.get("name", "") for performer in item.get("artists", []))
    return FoundTrack(uri=item["uri"], artist=artists, title=item.get("name", ""))


def stored_secret() -> str:
    """The environment first (a terminal or CI run), else the Keychain, else nothing."""
    from_environment = os.environ.get(SECRET_ENV_VAR, "")
    if from_environment:
        return from_environment
    try:
        return keychain.get_secret() or ""
    except keychain.KeychainError:
        return ""


def save_credentials(client_id: str, client_secret: str, *, session: HttpClient | None = None) -> None:
    """Prove the credentials work (one token request), then keep them: the id in config.toml, the secret in the
    Keychain. Raises SpotifyAPIError when Spotify refuses them, and nothing is written."""
    from ..config import save_setting

    SpotifyAPI(client_id, client_secret, session=session).token()
    save_setting("spotify_client_id", client_id)
    keychain.set_secret(client_secret)
