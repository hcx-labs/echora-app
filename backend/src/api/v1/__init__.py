"""API v1: every category, mounted under ``/api/v1``.

Each category is a package whose ``CATEGORY`` lists its endpoint classes (one per file).
"""

from __future__ import annotations

from importlib import import_module

from fastapi import APIRouter

CATEGORIES = (
    "system",
    "auth",
    "settings",
    "sync",
    "library",
    "search",
    "curations",
    "jobs",
    "media",
    "plugin",
)
"""Mounted in this order; route matching is per category prefix."""


def router() -> APIRouter:
    """Every v1 endpoint."""
    api = APIRouter(prefix="/api/v1")
    for name in CATEGORIES:
        api.include_router(import_module(f"backend.api.v1.{name}").CATEGORY.router())
    return api
