"""Exact URI lookup through the Spotify Web API (client-credentials: an app id + secret, no user login).

Optional. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (a free app at developer.spotify.com) and the resolver
stops depending on the LLM's memory for ids — which matters most with local models. Still validated by oEmbed.
"""
from __future__ import annotations

import base64
import os
import re
import time

import requests

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


class SpotifySearch:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None, session: requests.Session | None = None):
        self.client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        self.http = session or requests.Session()
        self._token, self._expires = None, 0.0

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _bearer(self) -> str:
        if self._token and time.time() < self._expires - 30:
            return self._token
        auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        r = self.http.post(TOKEN_URL, data={"grant_type": "client_credentials"}, headers={"Authorization": f"Basic {auth}"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        self._token, self._expires = data["access_token"], time.time() + float(data.get("expires_in", 3600))
        return self._token

    def __call__(self, artist: str, title: str) -> str | None:
        """Searcher signature for brain.resolve: the best-matching track's URI, or None."""
        if not self.configured:
            return None
        r = self.http.get(SEARCH_URL, params={"q": f"track:{title} artist:{artist}", "type": "track", "limit": 5},
                          headers={"Authorization": f"Bearer {self._bearer()}"}, timeout=15)
        if r.status_code != 200:
            return None
        items = r.json().get("tracks", {}).get("items", [])
        a, t = _norm(artist), _norm(title)

        def score(it):
            names = [_norm(x["name"]) for x in it.get("artists", [])]
            name = _norm(it.get("name", ""))
            return (any(a == x or a in x or x in a for x in names), t == name or t in name or name in t, it.get("popularity", 0))

        best = max(items, key=score, default=None)
        if best is None or not score(best)[0]:
            return None
        return best.get("uri")
