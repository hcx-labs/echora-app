"""Chromaprint fingerprints, and grouping songs that are the same recording."""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
import zlib

import numpy as np
import psycopg

from core.library import store
from worker.library.config import LibraryConfig

_MATCH_THRESHOLD = 0.88
_DURATION_TOLERANCE_SECONDS = 3.0
_MAX_ALIGNMENT_OFFSET = 10
_MIN_OVERLAP = 50
_MATCHER_REVISION = 1

_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
)


def compute_chromaprint(audio: bytes) -> tuple[bytes, float, str]:
    with tempfile.NamedTemporaryFile(suffix=".audio") as source:
        source.write(audio)
        source.flush()
        process = subprocess.run(
            ["fpcalc", "-raw", "-json", source.name],
            capture_output=True,
            text=True,
            timeout=LibraryConfig.load().fingerprint_timeout_seconds,
            check=False,
        )
    if process.returncode != 0:
        raise ValueError(f"fpcalc failed: {process.stderr.strip()[-400:]}")
    payload = json.loads(process.stdout)
    values = np.asarray(payload.get("fingerprint") or [], dtype=np.uint32)
    if not len(values):
        raise ValueError("fpcalc returned an empty fingerprint")
    return (
        zlib.compress(values.tobytes()),
        float(payload.get("duration") or 0),
        "chromaprint-fpcalc",
    )


def _decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(bytes(blob)), dtype=np.uint32)


def fingerprint_similarity(left_blob: bytes, right_blob: bytes) -> float | None:
    left, right = _decode(left_blob), _decode(right_blob)
    best: float | None = None
    for offset in range(-_MAX_ALIGNMENT_OFFSET, _MAX_ALIGNMENT_OFFSET + 1):
        if offset >= 0:
            first, second = left[offset:], right[: len(left) - offset]
        else:
            first, second = left[: len(right) + offset], right[-offset:]
        overlap = min(len(first), len(second))
        if overlap < _MIN_OVERLAP:
            continue
        differing = int(
            _POPCOUNT[(first[:overlap] ^ second[:overlap]).view(np.uint8)].sum(dtype=np.int64)
        )
        score = 1.0 - differing / (32.0 * overlap)
        best = score if best is None else max(best, score)
    return best


def store_and_match_fingerprint(
    connection: psycopg.Connection,
    track_id: uuid.UUID,
    audio: bytes,
    metadata_duration: float,
) -> dict[str, object]:
    """Fingerprint a song's audio, store it, and group it with a recording it matches."""
    fingerprint, measured_duration, version = compute_chromaprint(audio)
    return store_fingerprint(
        connection, track_id, fingerprint, measured_duration or metadata_duration, version
    )


def store_fingerprint(
    connection: psycopg.Connection,
    track_id: uuid.UUID,
    fingerprint: bytes,
    duration: float,
    version: str,
) -> dict[str, object]:
    """Store a computed fingerprint and group the song with a recording it matches."""
    store.save_fingerprint(connection, track_id, version, fingerprint, duration)
    candidates = store.fingerprint_candidates(
        connection,
        track_id,
        duration - _DURATION_TOLERANCE_SECONDS,
        duration + _DURATION_TOLERANCE_SECONDS,
    )
    matches: list[tuple[uuid.UUID, float, float]] = []
    for candidate in candidates:
        score = fingerprint_similarity(fingerprint, candidate["fingerprint"])
        if score is None:
            continue
        duration_delta = abs(duration - float(candidate["duration_seconds"]))
        if score >= _MATCH_THRESHOLD:
            matches.append((candidate["track_id"], score, duration_delta))
    if not matches:
        return {"matched": False, "candidates": len(candidates)}
    candidate_id, score, duration_delta = max(matches, key=lambda item: item[1])
    left, right = sorted((track_id, candidate_id))
    semantic = store.embedding_similarity(connection, left, right, "muq_mulan")
    acoustic = store.embedding_similarity(connection, left, right, "mert")
    evidence_id = store.save_match_evidence(
        connection,
        left,
        right,
        score,
        duration_delta,
        semantic,
        acoustic,
        _MATCHER_REVISION,
        {"threshold": _MATCH_THRESHOLD, "duration_tolerance_seconds": _DURATION_TOLERANCE_SECONDS},
    )
    group_id = store.group_of(connection, candidate_id)
    if group_id is None:
        group_id = store.add_recording_group(connection, candidate_id)
        store.add_group_member(connection, group_id, candidate_id, 1)
    store.add_group_member(connection, group_id, track_id, score)
    return {
        "matched": True,
        "group_id": str(group_id),
        "evidence_id": str(evidence_id),
        "confidence": score,
    }
