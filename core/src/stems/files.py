"""Where karaoke stems files live: relative paths under ``ECHORA_KARAOKE_STEMS_DIR``."""

from __future__ import annotations

from pathlib import Path

from core.media.config import KaraokeStemsConfig


def stems_root() -> Path:
    """The stems directory."""
    return Path(KaraokeStemsConfig.load().dir).resolve()


def stems_path(relative: str) -> Path:
    """Resolve a stored relative path, refusing anything outside the stems directory."""
    root = stems_root()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Stems path escapes the stems directory")
    return path
