"""The settings remote computation uses: the calling worker's own, so results match local ones.

Every config class with ``compute()`` fields takes part. A Modal call carries
``compute_settings()``; the container adopts it with ``adopt_compute_settings()`` before computing.
"""

from __future__ import annotations

from typing import Any

from core.config import EchoraConfig
from core.embeddings.config import EmbeddingConfig
from core.media.config import MotionArtworkConfig
from worker.kara.config import KaraConfig
from worker.transcription.config import TranscriptionConfig

COMPUTE_CONFIGS: tuple[type[EchoraConfig], ...] = (EmbeddingConfig, TranscriptionConfig, KaraConfig)


def compute_settings(*, motion_artwork_models: bool | None = None) -> dict[str, Any]:
    """This worker's compute fields by config class, and whether artwork models are needed.

    ``motion_artwork_models`` defaults to this worker's own setting; remote storage follows the
    Modal switch for motion artwork instead.
    """
    if motion_artwork_models is None:
        motion_artwork_models = MotionArtworkConfig.load().models
    return {
        "configs": {cls.__name__: cls.load().compute_values() for cls in COMPUTE_CONFIGS},
        "motion_artwork_models": bool(motion_artwork_models),
    }


def adopt_compute_settings(settings: dict[str, Any]) -> None:
    """On a remote container: compute with exactly the calling worker's settings."""
    unknown = set(settings) - {"configs", "motion_artwork_models"}
    configs = settings.get("configs", {})
    unknown |= {f"configs.{name}" for name in set(configs) - {c.__name__ for c in COMPUTE_CONFIGS}}
    if unknown:
        raise ValueError(f"Not compute settings: {sorted(unknown)}")
    for cls in COMPUTE_CONFIGS:
        cls.adopt(cls.with_compute_values(configs.get(cls.__name__, {})))
    if "motion_artwork_models" in settings:
        artwork = MotionArtworkConfig.load()
        MotionArtworkConfig.adopt(
            artwork.model_copy(update={"models": bool(settings["motion_artwork_models"])})
        )
