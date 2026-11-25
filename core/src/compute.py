"""Where the running job computes: on this server ("local") or on Modal ("modal").

The worker sets it for each job (``computing_on``); feature switches and planning read it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_location: ContextVar[str] = ContextVar("echora_compute_location", default="local")


def location() -> str:
    """Where the running job computes: "local" or "modal". Feature switches depend on it."""
    return _location.get()


@contextmanager
def computing_on(where: str) -> Iterator[None]:
    """Run a job's planning and stages for one location (the sync's choice)."""
    if where not in {"local", "modal"}:
        raise ValueError(f"Unknown compute location: {where}")
    token = _location.set(where)
    try:
        yield
    finally:
        _location.reset(token)
