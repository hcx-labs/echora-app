from __future__ import annotations

from worker.library.config import LibraryConfig

import json
import subprocess
import tempfile
import uuid
import zlib

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

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


def _embedding_similarity(
    cursor: psycopg.Cursor, left: uuid.UUID, right: uuid.UUID, model: str
) -> float | None:
    cursor.execute(
        """
        WITH latest AS (
          SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding
          FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
          WHERE e.track_id=ANY(%s) AND e.embedding_type='audio-track' AND ar.model_name=%s
          ORDER BY e.track_id, ar.created_at DESC
        )
        SELECT 1 - (a.embedding <=> b.embedding)
        FROM latest a JOIN latest b ON a.track_id=%s AND b.track_id=%s
        """,
        ([left, right], model, left, right),
    )
    row = cursor.fetchone()
    return float(row[0]) if row and row[0] is not None else None


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
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO track_fingerprints
                 (track_id, algorithm, algorithm_version, fingerprint, duration_seconds)
               VALUES (%s, 'chromaprint', %s, %s, %s)
               ON CONFLICT (track_id) DO UPDATE SET fingerprint=EXCLUDED.fingerprint,
                 algorithm_version=EXCLUDED.algorithm_version, duration_seconds=EXCLUDED.duration_seconds""",
            (track_id, version, fingerprint, duration),
        )
        cursor.execute(
            """SELECT track_id, fingerprint, duration_seconds FROM track_fingerprints
               WHERE track_id<>%s AND duration_seconds BETWEEN %s AND %s""",
            (
                track_id,
                duration - _DURATION_TOLERANCE_SECONDS,
                duration + _DURATION_TOLERANCE_SECONDS,
            ),
        )
        candidates = cursor.fetchall()
        matches: list[tuple[uuid.UUID, float, float]] = []
        for candidate_id, candidate_fingerprint, candidate_duration in candidates:
            score = fingerprint_similarity(fingerprint, candidate_fingerprint)
            if score is None:
                continue
            duration_delta = abs(duration - float(candidate_duration))
            if score >= _MATCH_THRESHOLD:
                matches.append((candidate_id, score, duration_delta))
        if not matches:
            return {"matched": False, "candidates": len(candidates)}
        candidate_id, score, duration_delta = max(matches, key=lambda item: item[1])
        left, right = sorted((track_id, candidate_id))
        semantic = _embedding_similarity(cursor, left, right, "muq_mulan")
        acoustic = _embedding_similarity(cursor, left, right, "mert")
        cursor.execute(
            """INSERT INTO recording_match_evidence
                 (left_track_id, right_track_id, decision, chromaprint_score,
                  duration_delta_seconds, semantic_similarity, acoustic_similarity,
                  matcher_revision, evidence)
               VALUES (%s,%s,'matched',%s,%s,%s,%s,%s,%s)
               ON CONFLICT (left_track_id, right_track_id, matcher_revision)
               DO UPDATE SET decision='matched', chromaprint_score=EXCLUDED.chromaprint_score,
                 duration_delta_seconds=EXCLUDED.duration_delta_seconds,
                 semantic_similarity=EXCLUDED.semantic_similarity,
                 acoustic_similarity=EXCLUDED.acoustic_similarity
               RETURNING id""",
            (
                left,
                right,
                score,
                duration_delta,
                semantic,
                acoustic,
                _MATCHER_REVISION,
                Jsonb(
                    {
                        "threshold": _MATCH_THRESHOLD,
                        "duration_tolerance_seconds": _DURATION_TOLERANCE_SECONDS,
                    }
                ),
            ),
        )
        evidence_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT group_id FROM recording_group_members WHERE track_id=%s", (candidate_id,)
        )
        existing = cursor.fetchone()
        if existing:
            group_id = existing[0]
        else:
            cursor.execute(
                "INSERT INTO recording_groups (canonical_track_id) VALUES (%s) RETURNING id",
                (candidate_id,),
            )
            group_id = cursor.fetchone()[0]
            cursor.execute(
                """INSERT INTO recording_group_members (group_id, track_id, confidence)
                   VALUES (%s,%s,1) ON CONFLICT (track_id) DO NOTHING""",
                (group_id, candidate_id),
            )
        cursor.execute(
            """INSERT INTO recording_group_members (group_id, track_id, confidence)
               VALUES (%s,%s,%s) ON CONFLICT (track_id) DO NOTHING""",
            (group_id, track_id, score),
        )
    return {
        "matched": True,
        "group_id": str(group_id),
        "evidence_id": str(evidence_id),
        "confidence": score,
    }
