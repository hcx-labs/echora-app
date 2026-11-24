"""Lyrics language detection models."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class LanguageConfig(EchoraConfig, env_prefix="ECHORA_LANGUAGE_"):
    """Language identification model files."""

    lid_model_path: str = setting("/models/lid.176.bin", legacy="LID_MODEL_PATH")
    indiclid_ftr_path: str = setting("/models/indiclid-ftr.bin", legacy="INDICLID_FTR_PATH")
