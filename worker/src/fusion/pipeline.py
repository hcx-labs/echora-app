"""Semantic fusion: one representation from the lyrics and audio track vectors.

Mirrors the corpus-whitened, weighted-concatenation approach used for hybrid
lyric+audio similarity: each modality is L2-normalized upstream, whitened with
per-dimension statistics computed over the paired corpus, then both sides are
scaled by the square root of their configured weight and concatenated. The
square root keeps each weight equal to its share of squared distance.

The whitening statistics and weights live in the analysis run config, so each
rebuild publishes a new representation contract and `current_embeddings` never
mixes generations.
"""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from collections.abc import Callable

import numpy as np
import psycopg

from core.db.connection import connect
from core.embeddings import store
from core.embeddings.representations import (
    configure_representations,
    model_settings,
)
from worker.fusion.config import FusionConfig

MODEL_NAME = "semantic_fusion"
MODEL_REVISION = "fusion-v1"
EMBEDDING_TYPE = "semantic_fusion"
AGGREGATION = "weighted_whitened_normalized_concat"
# Guards against divide-by-zero on dimensions that never vary.
STD_FLOOR = 1e-6


def fusion_weights() -> tuple[float, float]:
    lyrics = float(FusionConfig.load().weight_lyrics)
    audio = float(FusionConfig.load().weight_audio)
    total = lyrics + audio
    return lyrics / total, audio / total


def _vector(value) -> np.ndarray:
    if isinstance(value, str):
        return np.fromstring(value.strip("[]"), sep=",", dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _paired_vectors(
    connection: psycopg.Connection,
) -> tuple[list[uuid.UUID], np.ndarray, np.ndarray]:
    """Track-level lyrics (bge_m3) and audio (muq_mulan) vectors for tracks with both."""
    rows = store.fusion_pairs(connection)
    track_ids = [row["track_id"] for row in rows]
    lyrics = np.array([_vector(row["lyrics"]) for row in rows])
    audio = np.array([_vector(row["audio"]) for row in rows])
    return track_ids, lyrics, audio


def _fuse(
    lyrics: np.ndarray, audio: np.ndarray, weights: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lyrics_scale = np.sqrt(weights[0])
    audio_scale = np.sqrt(weights[1])
    lyrics_mean = lyrics.mean(axis=0)
    lyrics_std = np.maximum(lyrics.std(axis=0), STD_FLOOR)
    audio_mean = audio.mean(axis=0)
    audio_std = np.maximum(audio.std(axis=0), STD_FLOOR)
    lyrics_whitened = (lyrics - lyrics_mean) / lyrics_std * lyrics_scale
    audio_whitened = (audio - audio_mean) / audio_std * audio_scale
    fused = np.concatenate([lyrics_whitened, audio_whitened], axis=1)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    return (
        fused / np.maximum(norms, STD_FLOOR),
        np.concatenate([lyrics_mean, audio_mean]),
        np.concatenate([lyrics_std, audio_std]),
    )


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _run_config(
    weights: tuple[float, float],
    mean: np.ndarray,
    std: np.ndarray,
    lyrics_revision: str,
    audio_revision: str,
    track_count: int,
) -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "lyrics_model": "bge_m3",
        "lyrics_revision": lyrics_revision,
        "audio_model": "muq_mulan",
        "audio_revision": audio_revision,
        "weights": {"lyrics": weights[0], "audio": weights[1]},
        "aggregation": AGGREGATION,
        "whitening": "per_dimension_corpus",
        "track_count": track_count,
        "mean": [round(float(v), 8) for v in mean],
        "std": [round(float(v), 8) for v in std],
    }


def build_semantic_fusion(
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, int]:
    report = progress or (lambda _: None)
    weights = fusion_weights()
    _, lyrics_revision = model_settings("bge_m3")
    _, audio_revision = model_settings("muq_mulan")
    with connect() as connection:
        configure_representations(connection)
        report({"phase": "loading", "message": "Loading paired lyrics and audio embeddings"})
        track_ids, lyrics, audio = _paired_vectors(connection)
        if not track_ids:
            report(
                {
                    "phase": "writing",
                    "message": "No paired audio and lyrics embeddings to fuse",
                    "completed": 0,
                    "total": 0,
                    "unit": "vectors",
                }
            )
            return {"total": 0, "fused": 0}
        fused, mean, std = _fuse(lyrics, audio, weights)
        config = _run_config(weights, mean, std, lyrics_revision, audio_revision, len(track_ids))
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        run_id = store.replace_fusion_run(
            connection,
            EMBEDDING_TYPE,
            MODEL_NAME,
            MODEL_REVISION,
            config_hash,
            config,
            {"python": platform.python_version()},
        )
        report(
            {
                "phase": "writing",
                "message": f"Storing {len(track_ids)} fused vectors",
                "completed": 0,
                "total": len(track_ids),
                "unit": "vectors",
            }
        )
        store.save_fusion(
            connection,
            [
                (
                    track_ids[i],
                    run_id,
                    EMBEDDING_TYPE,
                    fused.shape[1],
                    AGGREGATION,
                    _vector_literal(fused[i]),
                )
                for i in range(len(track_ids))
            ],
            MODEL_NAME,
            MODEL_REVISION,
            config_hash,
            fused.shape[1],
            config,
        )
        connection.commit()
    report(
        {
            "phase": "writing",
            "message": f"Stored {len(track_ids)} fused vectors",
            "completed": len(track_ids),
            "total": len(track_ids),
            "unit": "vectors",
        }
    )
    return {"total": len(track_ids), "fused": len(track_ids)}
