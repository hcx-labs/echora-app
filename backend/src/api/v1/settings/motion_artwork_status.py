"""``GET /settings/motion-artwork/status``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.settings._deps import admin_required
from worker.artwork.config import ComfyUIConfig
from worker.artwork.render import MODEL_FILES, REQUIRED_LOADERS
from worker.artwork.settings import _connect, load_settings


class ExternalAIState(ResponseModel):
    """Whether External AI can write prompts, and its model."""

    enabled: bool
    model: str | None


class ArtworkCounts(ResponseModel):
    """Rendered and failed loops across the instance, and the bytes they take."""

    complete: int
    failed: int
    bytes: int


class MotionArtworkState(ResponseModel):
    """Whether motion artwork can render, and what it has rendered."""

    enabled: bool
    comfyui: dict[str, Any]
    """Embedded: ``installed`` and ``missing_models``. External: ``url``, ``reachable``,
    ``version``, ``device`` and ``missing_models``."""
    external_ai: ExternalAIState
    artworks: ArtworkCounts
    tracks_with_artwork: int
    """The user's songs with a loop."""


def _embedded_status() -> dict[str, object]:
    from huggingface_hub import try_to_load_from_cache

    comfyui = ComfyUIConfig.load()
    installed = (Path(comfyui.dir) / "main.py").is_file() and Path(comfyui.python).is_file()
    missing = [
        filename.rsplit("/", 1)[-1]
        for repo, revision, filename in MODEL_FILES
        if not isinstance(try_to_load_from_cache(repo, filename, revision=revision), str)
    ]
    return {"mode": "embedded", "installed": installed, "missing_models": missing}


def _external_status(url: str) -> dict[str, object]:
    from worker.artwork.comfyui import ComfyUI

    status: dict[str, object] = {
        "mode": "external",
        "url": url,
        "reachable": False,
        "missing_models": [],
    }
    try:
        with ComfyUI(url, timeout_seconds=5) as client:
            stats = client.system_stats()
            devices = stats.get("devices") or [{}]
            status.update(
                reachable=True,
                version=stats.get("system", {}).get("comfyui_version"),
                device=devices[0].get("name"),
            )
            status["missing_models"] = [
                name
                for node, field, name in REQUIRED_LOADERS
                if name not in client.available(node, field)
            ]
    except Exception:  # noqa: BLE001 - unreachable is a normal state while that ComfyUI is stopped
        pass
    return status


class MotionArtworkStatus(Endpoint):
    """ComfyUI's state, External AI's readiness and the loop counts (administrators only)."""

    method = "GET"
    path = "/motion-artwork/status"
    response = MotionArtworkState

    def handle(self, user: SessionUser = Depends(admin_required)) -> dict[str, object]:
        with _connect() as connection, connection.cursor() as cursor:
            settings = load_settings(cursor)
            cursor.execute(
                """SELECT count(*) FILTER (WHERE status='complete') AS complete,
                          count(*) FILTER (WHERE status='failed') AS failed,
                          coalesce(sum(bytes) FILTER (WHERE status='complete'), 0) AS bytes
                   FROM motion_artworks"""
            )
            counts = cursor.fetchone()
            cursor.execute(
                """SELECT count(DISTINCT utl.track_id) AS tracks FROM user_track_links utl
                   JOIN track_motion_artwork tma ON tma.track_id = utl.track_id
                   JOIN motion_artworks ma ON ma.cover_sha256 = tma.cover_sha256
                                          AND ma.status='complete'
                   WHERE utl.user_id = %s""",
                (user["id"],),
            )
            tracks = cursor.fetchone()["tracks"]
        url = settings.external_url()
        from core.lyrics.translations import load_settings as load_external_ai

        external_ai, _ = load_external_ai()
        return {
            "enabled": settings.enabled,
            "comfyui": _external_status(url) if url else _embedded_status(),
            "external_ai": {
                "enabled": external_ai.ready(),
                "model": external_ai.model or None,
            },
            "artworks": dict(counts),
            "tracks_with_artwork": tracks,
        }
