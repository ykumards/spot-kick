"""Track search through the Spotify Web API with the developer's own app credentials.

The Client Credentials flow exchanges the client id (config.toml) and secret (Keychain, or the environment for
terminal runs) for an app token, cached until it expires. No user login is involved and only ``/v1/search`` is
used. Each developer registers their own app; no credentials are shipped, and the secret is never written to a
file.
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
    """The ``requests`` module, a ``Session``, or a test fake."""

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
        """Replace the credentials in place; the next call fetches a fresh token."""
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._token_expires_at = 0.0

    def token(self) -> str:
        """Return the app token, refreshing it TOKEN_SAFETY_MARGIN_S before it expires."""
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
        """Return Spotify's search results for the name, in its order. Raises SpotifyAPIError on failure."""
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
    """Return the client secret from the environment, else the Keychain, else an empty string."""
    from_environment = os.environ.get(SECRET_ENV_VAR, "")
    if from_environment:
        return from_environment
    try:
        return keychain.get_secret() or ""
    except keychain.KeychainError:
        return ""


def save_credentials(client_id: str, client_secret: str, *, session: HttpClient | None = None) -> None:
    """Validate the credentials with one token request, then store them.

    Raises SpotifyAPIError when Spotify refuses them; nothing is written in that case.
    """
    from ..config import save_setting

    SpotifyAPI(client_id, client_secret, session=session).token()
    save_setting("spotify_client_id", client_id)
    keychain.set_secret(client_secret)
