"""Melody index configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class MelodyConfig(EchoraConfig, env_prefix="ECHORA_HUM_"):
    """Hum search corpus building and queries."""

    max_workers: int = setting(16, gt=0, description="Threads scoring one query.")
    diagnostic_dir: str = setting("/models/torch/hum-diagnostics", legacy="HUM_DIAGNOSTIC_DIR")
