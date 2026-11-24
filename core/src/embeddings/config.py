"""Embedding models: which checkpoints compute the stored vectors."""

from __future__ import annotations

from core.config import EchoraConfig, compute, setting


class EmbeddingConfig(EchoraConfig, env_prefix="ECHORA_EMBEDDING_"):
    """Audio (MuQ-MuLan, MERT) and lyrics (BGE-M3) embedding models, pinned by revision."""

    muq_model_id: str = compute("OpenMuQ/MuQ-MuLan-large", legacy="MUQ_MODEL_ID")
    muq_revision: str = compute("2e01c796b71dca71b45251384c04cd7b237c9020", legacy="MUQ_REVISION")
    mert_model_id: str = compute("m-a-p/MERT-v1-95M", legacy="MERT_MODEL_ID")
    mert_revision: str = compute("12af15fef9d0ac838c3f475bfbbf26d2060dd4f5", legacy="MERT_REVISION")
    lyrics_model_id: str = compute("BAAI/bge-m3", legacy="LYRICS_MODEL_ID")
    lyrics_revision: str = compute(
        "5617a9f61b028005a4858fdac845db406aefb181", legacy="LYRICS_REVISION"
    )
    lyrics_window_batch_size: int = setting(
        2, gt=0, legacy="ECHORA_LYRICS_WINDOW_BATCH_SIZE", description="Lyric windows per batch."
    )
    lyrics_track_batch_size: int = setting(
        8, gt=0, legacy="ECHORA_LYRICS_TRACK_BATCH_SIZE", description="Songs per lyrics batch."
    )
