"""Small Reddit HTTP client with application-only OAuth and public fallback."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import requests


class RedditClient:
    """Fetch Reddit listings with a cached client-credentials token when configured."""

    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    OAUTH_BASE = "https://oauth.reddit.com"
    PUBLIC_BASE = "https://www.reddit.com"

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None,
                 user_agent: str = "stock-analysis-bot/1.0", session: Any = None,
                 clock: Callable[[], float] = time.time) -> None:
        if bool(client_id) != bool(client_secret):
            raise ValueError("both REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are required")
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.session = session or requests.Session()
        self.clock = clock
        self._access_token: Optional[str] = None
        self._token_expires_at = 0.0

    @classmethod
    def from_env(cls) -> "RedditClient":
        return cls(os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET"),
                   os.getenv("REDDIT_USER_AGENT", "stock-analysis-bot/1.0"))

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def degraded_public_fallback(self) -> bool:
        return not self.authenticated

    def _token(self, timeout: int = 15) -> str:
        now = self.clock()
        if self._access_token and now < self._token_expires_at:
            return self._access_token
        response = self.session.post(
            self.TOKEN_URL, auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"}, headers={"User-Agent": self.user_agent},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = payload["access_token"]
        self._token_expires_at = now + max(0, int(payload.get("expires_in", 3600)) - 60)
        return self._access_token

    @staticmethod
    def _path(value: str) -> str:
        parsed = urlparse(value)
        path = parsed.path if parsed.scheme else value
        if path.endswith(".json"):
            path = path[:-5]
        return "/" + path.lstrip("/")

    def get(self, path: str, *, params: Optional[dict[str, Any]] = None, timeout: int = 15):
        normalized = self._path(path)
        if self.authenticated:
            headers = {"Authorization": f"bearer {self._token(timeout)}", "User-Agent": self.user_agent}
            url = f"{self.OAUTH_BASE}{normalized}"
            response = self.session.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code == 401:
                self._access_token = None
                self._token_expires_at = 0.0
                headers["Authorization"] = f"bearer {self._token(timeout)}"
                response = self.session.get(url, headers=headers, params=params, timeout=timeout)
            return response
        else:
            headers = {"User-Agent": self.user_agent}
            url = f"{self.PUBLIC_BASE}{normalized}.json"
        return self.session.get(url, headers=headers, params=params, timeout=timeout)
