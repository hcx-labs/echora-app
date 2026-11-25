"""``GET /settings/runtime``."""

from __future__ import annotations

from fastapi import Response

from backend.api.endpoint import Endpoint
from backend.config import UiConfig


class RuntimeSettings(Endpoint):
    """The browser's polling intervals and timeouts (``UiConfig``); no session needed."""

    method = "GET"
    path = "/runtime"
    response = dict[str, int]

    def handle(self, response: Response) -> dict[str, int]:
        # Only this explicit, non-secret allowlist is browser-visible.
        response.headers["Cache-Control"] = "no-store"
        return UiConfig.load().public_ui()
