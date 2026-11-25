"""Navidrome credentials as a request body, shared by settings and sync."""

from __future__ import annotations

from pydantic import Field, HttpUrl, SecretStr

from backend.api.requests import PostRequest


class Credentials(PostRequest):
    """A Navidrome server and an account on it."""

    url: HttpUrl
    username: str = Field(min_length=1)
    password: SecretStr

    def as_tuple(self) -> tuple[str, str, str]:
        """``(url, username, password)`` with the trailing slash removed from the URL."""
        return str(self.url).rstrip("/"), self.username, self.password.get_secret_value()
