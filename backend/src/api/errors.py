"""Domain exceptions as HTTP responses, in one place.

Messages of these exceptions are written for the user and safe to show; any other
``EchoraError`` stays a server error.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.errors import (
    ConflictError,
    EchoraError,
    IntegrationError,
    InvalidQueryError,
    NotFoundError,
)

ERROR_STATUS = (
    (NotFoundError, 404),
    (ConflictError, 409),
    (InvalidQueryError, 422),
    (IntegrationError, 502),
)


async def _echora_error(request: Request, error: EchoraError) -> JSONResponse:
    for kind, status in ERROR_STATUS:
        if isinstance(error, kind):
            return JSONResponse({"detail": str(error)}, status_code=status)
    raise error


def install_error_handlers(app: FastAPI) -> None:
    """Map domain exceptions to their HTTP status."""
    app.add_exception_handler(EchoraError, _echora_error)
