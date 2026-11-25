"""Motion artwork settings, the selection of songs to render, and loop lookup.

Loops are rendered by the analysis worker as a stage of sync and import batches. Administrators
configure generation; every signed-in user can play loops for tracks in their own library (the
endpoints are in the API). Videos live under ECHORA_MOTION_ARTWORK_DIR, separate
from model storage.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, field_validator, model_validator

from core.db.config import DatabaseConfig
from core.media.config import MotionArtworkConfig
from worker.artwork.config import ComfyUIConfig
from worker.artwork.render import (
    CAMERA_LORA_STRENGTH,
    DEFAULT_INSTRUCTIONS,
    Recipe,
)

FIELDS = (
    "enabled",
    "comfyui_url",
    "resolution",
    "frames",
    "upscale",
    "camera_lock",
    "prompt_mode",
    "instructions",
    "fixed_prompt",
    "seed",
    "generate_during_sync",
    "generate_on_modal",
    "regenerate_outdated",
)


class MotionArtworkSettings(BaseModel):
    """Motion artwork settings as stored, with their defaults."""

    enabled: bool = False
    # Blank starts the worker's embedded ComfyUI (or ECHORA_COMFYUI_URL when set).
    comfyui_url: str = Field("", max_length=500)
    # The size LTX renders at; upscaling doubles it (768 renders a 1536 loop).
    resolution: Literal[512, 768, 1024, 1536] = 768
    # 5 or 10 seconds at 24 fps; the cover is the first and last frame.
    frames: Literal[121, 241] = 241
    upscale: bool = True
    # Strength of Lightricks' static-camera LoRA: higher holds the framing harder but damps motion.
    camera_lock: float = Field(CAMERA_LORA_STRENGTH, ge=0, le=1)
    # "external" asks the configured External AI model; "auto" uses the built-in Gemma 4 E2B.
    prompt_mode: Literal["auto", "external", "fixed"] = "auto"
    # Blank means the built-in instructions.
    instructions: str = Field("", max_length=4000)
    fixed_prompt: str = Field("", max_length=4000)
    # None picks a new seed for every album.
    seed: int | None = Field(42, ge=0, le=2**53)
    # Render missing loops in syncs on this server, and in syncs on Modal (docs/modal-compute.md).
    generate_during_sync: bool = True
    generate_on_modal: bool = False
    # Off: changed settings apply to loops rendered from then on and existing loops stay.
    # On: syncs render loops made with other settings again.
    regenerate_outdated: bool = False

    @field_validator("comfyui_url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return value
        parts = urlsplit(value)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("ComfyUI URL must be an http(s) address without credentials")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _fixed_prompt(self):
        if self.prompt_mode == "fixed" and not self.fixed_prompt.strip():
            raise ValueError("A fixed prompt is required in fixed prompt mode")
        if self.upscale and self.resolution > 768:
            raise ValueError("Upscaling renders at 512 or 768 and doubles it")
        return self

    def recipe(self) -> Recipe:
        # Gemma needs full instructions; External AI has a built-in system prompt and takes extras.
        instructions = self.instructions.strip()
        if self.prompt_mode == "auto":
            instructions = instructions or DEFAULT_INSTRUCTIONS
        return Recipe(
            resolution=self.resolution,
            frames=self.frames,
            upscale=self.upscale,
            camera_lock=self.camera_lock,
            prompt_mode=self.prompt_mode,
            instructions=instructions,
            fixed_prompt=self.fixed_prompt.strip(),
            seed=self.seed,
        )

    def external_url(self) -> str:
        """A running ComfyUI to use instead of starting one, or blank for the embedded one."""
        return self.comfyui_url or ComfyUIConfig.load().url


def _connect():
    return psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row)


def load_settings(cursor) -> MotionArtworkSettings:
    cursor.execute(f"SELECT {', '.join(FIELDS)} FROM motion_artwork_settings WHERE singleton=true")
    row = cursor.fetchone()
    if row is None:
        return MotionArtworkSettings()
    if not isinstance(row, dict):
        row = dict(zip(FIELDS, row))
    return MotionArtworkSettings.model_validate(dict(row))


def save_settings(cursor, settings: MotionArtworkSettings) -> None:
    values = settings.model_dump()
    columns = ", ".join(FIELDS)
    placeholders = ", ".join(f"%({name})s" for name in FIELDS)
    updates = ", ".join(f"{name}=EXCLUDED.{name}" for name in FIELDS)
    cursor.execute(
        f"""INSERT INTO motion_artwork_settings (singleton, {columns}, updated_at)
            VALUES (true, {placeholders}, now())
            ON CONFLICT (singleton) DO UPDATE SET {updates}, updated_at=now()""",
        values,
    )


def missing_external_ids(
    cursor, library_id, external_ids: list[str], recipe: Recipe, regenerate_outdated: bool = False
) -> set[str]:
    """Sources whose cover has no complete loop, including covers not yet seen.

    With regenerate_outdated, a loop made with another recipe counts as missing too.
    """
    cursor.execute(
        """SELECT DISTINCT ts.external_id FROM track_sources ts
           LEFT JOIN track_motion_artwork tma ON tma.track_id = ts.track_id
           WHERE ts.library_id = %s AND ts.source_type = 'subsonic' AND ts.external_id = ANY(%s)
             AND ts.source_data->>'coverArt' IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM motion_artworks ma WHERE ma.cover_sha256 = tma.cover_sha256
                             AND (%s OR ma.recipe_hash = %s) AND ma.status = 'complete')""",
        (library_id, list(external_ids), not regenerate_outdated, recipe.hash()),
    )
    return {row["external_id"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()}


def sync_selection(connection, library_id, external_ids: list[str]) -> set[str]:
    """Songs an entire-library sync should visit so their album gets a loop."""
    with connection.cursor(row_factory=dict_row) as cursor:
        settings = load_settings(cursor)
        from core.compute import location

        # The sync's location decides: syncs on this server or syncs on Modal.
        on_modal = location() == "modal"
        if not (
            settings.enabled
            and (settings.generate_on_modal if on_modal else settings.generate_during_sync)
        ):
            return set()
        return missing_external_ids(
            cursor, library_id, external_ids, settings.recipe(), settings.regenerate_outdated
        )


def clear_loops() -> dict[str, int]:
    """Delete every loop and its track mapping, then the video files. Refused while a sync runs."""
    with _connect() as db:
        # Block new sync jobs until the check and the deletion commit together.
        db.execute("LOCK TABLE jobs IN SHARE ROW EXCLUSIVE MODE")
        active = db.execute("""SELECT 1 FROM jobs WHERE status IN ('queued','running','waiting')
            AND (kind IN ('navidrome_sync','import')
                 OR (kind='analysis_batch' AND payload->>'operation' IN ('navidrome_sync','import')))
            LIMIT 1""").fetchone()
        if active:
            raise RuntimeError("sync_active")
        paths = [
            row["path"]
            for row in db.execute("DELETE FROM motion_artworks RETURNING path").fetchall()
            if row["path"]
        ]
        db.execute("DELETE FROM track_motion_artwork")
    # Files go after the commit, so a failure never leaves rows pointing at missing videos.
    removed = 0
    for relative in paths:
        try:
            artwork_path(relative).unlink(missing_ok=True)
            removed += 1
        except (OSError, ValueError):
            pass
    for directory in sorted(Path(MotionArtworkConfig.load().dir).glob("*/"), reverse=True):
        with contextlib.suppress(OSError):
            directory.rmdir()  # Only removes directories that are now empty.
    return {"deleted": len(paths), "files_removed": removed}


def artwork_path(relative: str) -> Path:
    """Resolve a stored relative path, refusing anything outside the artwork directory."""
    root = Path(MotionArtworkConfig.load().dir).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Artwork path escapes the artwork directory")
    return path


def track_artwork(cursor, user_id, track_id) -> dict | None:
    """The newest complete loop for a track's cover, preferring the current recipe."""
    settings = load_settings(cursor)
    cursor.execute(
        """SELECT ma.id, ma.width, ma.height, ma.frames, ma.fps, ma.prompt, ma.created_at
           FROM user_track_links utl
           JOIN track_motion_artwork tma ON tma.track_id = utl.track_id
           JOIN motion_artworks ma ON ma.cover_sha256 = tma.cover_sha256 AND ma.status = 'complete'
           WHERE utl.user_id = %s AND utl.track_id = %s
           ORDER BY (ma.recipe_hash = %s) DESC, ma.created_at DESC LIMIT 1""",
        (user_id, track_id, settings.recipe().hash()),
    )
    return cursor.fetchone()
