"""Motion artwork endpoints: settings and status (admin), and each user's loops and videos."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from worker.artwork.config import ComfyUIConfig
from worker.artwork.render import DEFAULT_INSTRUCTIONS, MODEL_FILES, REQUIRED_LOADERS
from worker.artwork.settings import (
    MotionArtworkSettings,
    _connect,
    artwork_path,
    clear_loops,
    load_settings,
    save_settings,
    track_artwork,
)


class LocationUpdate(BaseModel):
    enabled: bool
    location: Literal["local", "modal"] = "local"


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
    except Exception:
        # Unreachable is a normal state while that ComfyUI is stopped.
        pass
    return status


def router(require_user):
    api = APIRouter(tags=["motion-artwork"])

    def admin(user=Depends(require_user)):
        if not user.get("is_admin"):
            raise HTTPException(403, "Administrator access required")
        return user

    @api.get("/settings/motion-artwork")
    def get_settings_route(_=Depends(admin)) -> dict[str, object]:
        with _connect() as connection, connection.cursor() as cursor:
            settings = load_settings(cursor)
        return {
            **settings.model_dump(),
            "defaults": {
                "comfyui_url": ComfyUIConfig.load().url,
                "instructions": DEFAULT_INSTRUCTIONS,
            },
        }

    @api.put("/settings/motion-artwork")
    def put_settings_route(update: MotionArtworkSettings, _=Depends(admin)) -> dict[str, object]:
        with _connect() as connection, connection.cursor() as cursor:
            save_settings(cursor, update)
        return update.model_dump()

    @api.put("/settings/motion-artwork/location")
    def put_location_route(update: LocationUpdate, _=Depends(admin)) -> dict[str, object]:
        """Switch generation on or off for syncs on this server or on Modal."""
        column = "generate_on_modal" if update.location == "modal" else "generate_during_sync"
        with _connect() as connection, connection.cursor() as cursor:
            settings = load_settings(cursor).model_copy(update={column: update.enabled})
            save_settings(cursor, settings)
        return {
            "enabled": update.enabled,
            "location": update.location,
            "feature_enabled": settings.enabled,
        }

    @api.get("/settings/motion-artwork/status")
    def status_route(user=Depends(admin)) -> dict[str, object]:
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
                   JOIN motion_artworks ma ON ma.cover_sha256 = tma.cover_sha256 AND ma.status='complete'
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

    @api.delete("/settings/motion-artwork/loops")
    def clear_loops_route(_=Depends(admin)) -> dict[str, int]:
        try:
            return clear_loops()
        except RuntimeError:
            raise HTTPException(
                409, "Wait for active syncs and imports to finish before clearing loops"
            ) from None

    @api.get("/library/tracks/{track_id}/motion-artwork")
    def track_route(track_id: uuid.UUID, user=Depends(require_user)) -> dict[str, object]:
        with _connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
                (user["id"], track_id),
            )
            if cursor.fetchone() is None:
                raise HTTPException(404, "Track not found")
            artwork = track_artwork(cursor, user["id"], track_id)
        if artwork is None:
            return {"track_id": str(track_id), "available": False}
        return {
            "track_id": str(track_id),
            "available": True,
            "id": str(artwork["id"]),
            "url": f"/motion-artwork/{artwork['id']}.mp4",
            "width": artwork["width"],
            "height": artwork["height"],
            "frames": artwork["frames"],
            "fps": artwork["fps"],
        }

    @api.get("/motion-artwork/{artwork_id}.mp4")
    def video_route(artwork_id: uuid.UUID, user=Depends(require_user)) -> FileResponse:
        with _connect() as connection, connection.cursor() as cursor:
            # Only loops for covers in the caller's own library are served.
            cursor.execute(
                """SELECT ma.path FROM motion_artworks ma
                   WHERE ma.id = %s AND ma.status = 'complete' AND EXISTS (
                     SELECT 1 FROM track_motion_artwork tma
                     JOIN user_track_links utl ON utl.track_id = tma.track_id
                     WHERE tma.cover_sha256 = ma.cover_sha256 AND utl.user_id = %s)""",
                (artwork_id, user["id"]),
            )
            row = cursor.fetchone()
        if row is None:
            raise HTTPException(404, "Motion artwork not found")
        try:
            path = artwork_path(row["path"])
        except ValueError:
            raise HTTPException(404, "Motion artwork not found") from None
        if not path.is_file():
            raise HTTPException(404, "Motion artwork not found")
        # The file for an id never changes, so browsers may keep it.
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=604800, immutable"},
        )

    return api
