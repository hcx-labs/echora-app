"""Motion artwork settings as request and response bodies."""

from __future__ import annotations

from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from worker.artwork.settings import MotionArtworkSettings


class MotionArtworkUpdate(MotionArtworkSettings, PutRequest):
    """Every motion artwork setting."""


class MotionArtworkView(MotionArtworkSettings, ResponseModel):
    """Every motion artwork setting, as stored."""
