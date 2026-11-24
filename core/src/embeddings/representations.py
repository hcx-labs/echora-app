"""One configured representation contract per model, shared by writers and readers."""
from __future__ import annotations

import hashlib
import json
from core.settings import get_settings

from psycopg.types.json import Jsonb

MODELS = {
    "muq_mulan": ("MUQ", "OpenMuQ/MuQ-MuLan-large", "2e01c796b71dca71b45251384c04cd7b237c9020", 512),
    "mert": ("MERT", "m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5", 768),
    "bge_m3": ("LYRICS", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", 1024),
}


def model_settings(name: str) -> tuple[str, str]:
    prefix, model_id, revision, _ = MODELS[name]
    settings = get_settings()
    return getattr(settings, f"{prefix.lower()}_model_id"), getattr(settings, f"{prefix.lower()}_revision")


def config_hash(config: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def embedding_config(name: str, revision: str | None = None) -> dict[str, object]:
    model_id, configured_revision = model_settings(name)
    revision = revision or configured_revision
    if name == "bge_m3":
        config = {
            "model": name, "revision": revision, "chunk_tokens": 7168,
            "overlap_tokens": 512, "aggregation": "normalized_mean",
            "pooling": "bge-m3 dense", "maximum_tokens": 8192,
        }
    else:
        config = {
            "model": name, "revision": revision, "sample_rate": 24000,
            "coverage": "full-track", "windows": None, "window_seconds": 10,
            "stride_seconds": 5, "aggregation": "normalized_mean",
            "store_window_embeddings": True,
        }
    # Preserve compatibility with existing default-model runs. Custom repositories
    # must not reuse a representation merely because their revision strings match.
    if model_id != MODELS[name][1]:
        config["model_id"] = model_id
    return config


def configure_representations(connection) -> None:
    """Publish the deployment's contracts, never a per-track 'latest' fallback.

    Call at startup and before standalone backfills. A changed configuration may
    temporarily reduce coverage; old and new vector spaces must never be mixed.
    """
    from worker.profiles.pipeline import AUDIO_PROFILE_REVISION, DEFAULT_PARAMETERS, _profile_config

    specs = []
    for name, (_, _, _, dimension) in MODELS.items():
        _, revision = model_settings(name)
        kind = "lyrics_embedding" if name == "bge_m3" else "audio_embedding"
        config = embedding_config(name)
        specs.append((kind, name, revision, config_hash(config), dimension, Jsonb(config)))
    for name in ("muq_mulan", "mert"):
        config = _profile_config(name, DEFAULT_PARAMETERS)
        specs.append(("audio_profile", name, AUDIO_PROFILE_REVISION,
                      config_hash(config), None, Jsonb(config)))
    config = voice_config()
    specs.append(("voice_classification", "mtg-jamendo-voice-gender-v2", "joint-v2",
                  config_hash(config), 3, Jsonb(config)))
    with connection.cursor() as cursor:
        cursor.executemany(
            """INSERT INTO active_representation_specs
                 (kind, model_name, model_revision, config_hash, dimension, config)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (kind, model_name) DO UPDATE SET
                 model_revision=EXCLUDED.model_revision, config_hash=EXCLUDED.config_hash,
                 dimension=EXCLUDED.dimension, config=EXCLUDED.config""", specs,
        )


def voice_config() -> dict[str, object]:
    config = {
        "model": "mtg-jamendo-voice-gender-v2",
        "embedding": "discogs-effnet-bsdynamic-1",
        "labels": ["instrumental", "female", "male"],
        "aggregation": "mean_joint_activation",
        "preprocessing": "essentia-tensorflow-input-musicnn",
        "model_sha256": {
            "discogs-effnet-bsdynamic-1.onnx": "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c",
            "gender-discogs-effnet-1.onnx": "e3e865d4bf36d4817f32ddab9452b2729f9e33a4d068d1c44ea44972a7999e91",
            "voice_instrumental-discogs-effnet-1.onnx": "20155e4c439714b0c45c08644b73c8e12d9dccb173bd4ab9934bf1e5aee837ca",
        },
    }
    return config
