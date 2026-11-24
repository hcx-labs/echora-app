"""Semantic fusion configuration."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from core.config import EchoraConfig, setting


class FusionConfig(EchoraConfig, env_prefix="ECHORA_FUSION_"):
    """Weights of lyrics and audio in the fused track vector."""

    weight_lyrics: float = setting(0.75, ge=0, legacy="SEMANTIC_FUSION_WEIGHT_LYRICS")
    weight_audio: float = setting(0.25, ge=0, legacy="SEMANTIC_FUSION_WEIGHT_AUDIO")

    @model_validator(mode="after")
    def _positive_total(self) -> Self:
        if self.weight_lyrics + self.weight_audio <= 0:
            raise ValueError("Semantic fusion weights must have a positive total")
        return self
