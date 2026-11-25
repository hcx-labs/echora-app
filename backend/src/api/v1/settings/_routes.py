"""Settings routes whose validation errors never echo the request body."""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

# Model-level validation errors can echo the whole body, including API keys and tokens.
_REDACTED = (
    ("/settings/external-ai", "Invalid External AI settings"),
    ("/settings/external-processing", "Invalid External processing settings"),
)


class RedactedRoute(APIRoute):
    """Replaces validation errors of secret-bearing routes with a fixed message."""

    def get_route_handler(self):
        handler = super().get_route_handler()
        message = next((text for prefix, text in _REDACTED if prefix in self.path), None)
        if message is None:
            return handler

        async def redacted(request):
            try:
                return await handler(request)
            except RequestValidationError:
                raise HTTPException(422, message) from None

        return redacted
