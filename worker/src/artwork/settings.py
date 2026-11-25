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

from pydantic import BaseModel, Field, field_validator, model_validator

from core.artwork import store
from core.db.connection import connect
from core.media.config import MotionArtworkConfig
from worker.artwork.config import ComfyUIConfig
from worker.artwork.render import (
    CAMERA_LORA_STRENGTH,
    DEFAULT_INSTRUCTIONS,
    Recipe,
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


def load_settings(db) -> MotionArtworkSettings:
    """The stored settings, or the defaults before they are first saved."""
    row = store.settings(db)
    return MotionArtworkSettings.model_validate(row) if row else MotionArtworkSettings()


def save_settings(db, settings: MotionArtworkSettings) -> None:
    store.save_settings(db, settings.model_dump())


def missing_external_ids(
    db, library_id, external_ids: list[str], recipe: Recipe, regenerate_outdated: bool = False
) -> set[str]:
    """Sources whose cover has no complete loop, including covers not yet seen.

    With regenerate_outdated, a loop made with another recipe counts as missing too.
    """
    return store.sources_missing_loops(
        db, library_id, external_ids, recipe.hash(), regenerate_outdated
    )


def sync_selection(connection, library_id, external_ids: list[str]) -> set[str]:
    """Songs an entire-library sync should visit so their album gets a loop."""
    settings = load_settings(connection)
    from core.compute import location

    # The sync's location decides: syncs on this server or syncs on Modal.
    on_modal = location() == "modal"
    if not (
        settings.enabled
        and (settings.generate_on_modal if on_modal else settings.generate_during_sync)
    ):
        return set()
    return missing_external_ids(
        connection, library_id, external_ids, settings.recipe(), settings.regenerate_outdated
    )


def clear_loops() -> dict[str, int]:
    """Delete every loop and its track mapping, then the video files. Refused while a sync runs."""
    with connect() as db:
        if store.lock_out_syncs(db):
            raise RuntimeError("sync_active")
        paths = store.delete_all_loops(db)
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


def track_artwork(db, user_id, track_id) -> dict | None:
    """The newest complete loop for a track's cover, preferring the current recipe."""
    return store.track_loop(db, user_id, track_id, load_settings(db).recipe().hash())
