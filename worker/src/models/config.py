"""Model file locations."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class ModelsConfig(EchoraConfig, env_prefix="ECHORA_MODELS_"):
    """Where downloaded model files live, beyond the Hugging Face cache."""

    essentia_dir: str = setting("/data/models/essentia", legacy="ESSENTIA_MODELS_DIR")
