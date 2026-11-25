from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sklearn.metrics import adjusted_rand_score

from core.db.config import require_database_url
from core.embeddings.representations import (
    DEFAULT_PARAMETERS,
    AudioProfileParameters,
    profile_config,
)
from core.revisions import AUDIO_PROFILE_REVISION, SUPPORTED_PROFILE_MODELS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowVector:
    index: int
    start_seconds: float
    end_seconds: float
    vector: np.ndarray

    @property
    def center_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(frozen=True)
class TemporalSegment:
    index: int
    start_seconds: float
    end_seconds: float
    vector: np.ndarray
    cohesion: float
    representative_window_index: int


@dataclass(frozen=True)
class ModeInterval:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class MusicalMode:
    index: int
    vector: np.ndarray
    duration_weight: float
    cohesion: float
    representative_window_index: int
    intervals: tuple[ModeInterval, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrackAudioProfile:
    global_overlap: np.ndarray
    global_decorrelated: np.ndarray
    resultant_length: float
    mean_global_similarity: float
    p05_global_similarity: float
    adjacent_change_mean: float
    adjacent_change_p95: float
    timestamps_exact: bool
    segments: tuple[TemporalSegment, ...]
    modes: tuple[MusicalMode, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Cannot normalize an empty or invalid audio-profile vector")
    return value / norm


def _centroid(matrix: np.ndarray) -> np.ndarray:
    return _normalize(np.asarray(matrix, dtype=np.float32).mean(axis=0))


def infer_window_ranges(
    count: int,
    duration_seconds: float,
    window_seconds: float = 10.0,
    stride_seconds: float = 5.0,
) -> list[tuple[float, float]]:
    """Reconstruct legacy ranges using the same full-coverage rule as ingestion."""
    if count <= 0:
        return []
    if duration_seconds <= window_seconds or count == 1:
        return [(0.0, max(0.0, duration_seconds))]
    starts = [index * stride_seconds for index in range(count)]
    starts[-1] = max(0.0, duration_seconds - window_seconds)
    return [(start, min(duration_seconds, start + window_seconds)) for start in starts]


def _decorrelated_indices(windows: Sequence[WindowVector]) -> list[int]:
    """Greedily retain windows that do not overlap the last retained window."""
    selected: list[int] = []
    covered_until = -math.inf
    for index, window in enumerate(windows):
        if window.start_seconds >= covered_until - 1e-6:
            selected.append(index)
            covered_until = window.end_seconds
    return selected or [0]


def _interval_boundaries(windows: Sequence[WindowVector], duration_seconds: float) -> np.ndarray:
    centers = np.asarray([window.center_seconds for window in windows], dtype=np.float64)
    boundaries = np.empty(len(windows) + 1, dtype=np.float64)
    boundaries[0] = max(0.0, windows[0].start_seconds)
    boundaries[-1] = min(duration_seconds, windows[-1].end_seconds)
    if len(windows) > 1:
        boundaries[1:-1] = (centers[:-1] + centers[1:]) / 2
    return boundaries


def _segment_cost(prefix: np.ndarray, start: int, end: int) -> float:
    count = end - start
    return float(count - np.linalg.norm(prefix[end] - prefix[start]))


def _temporal_segments(
    windows: Sequence[WindowVector],
    duration_seconds: float,
    parameters: AudioProfileParameters,
) -> tuple[TemporalSegment, ...]:
    matrix = np.stack([window.vector for window in windows])
    count = len(windows)
    if count == 1:
        return (
            TemporalSegment(
                0,
                windows[0].start_seconds,
                windows[0].end_seconds,
                matrix[0],
                1.0,
                windows[0].index,
            ),
        )

    boundaries = _interval_boundaries(windows, duration_seconds)
    maximum = min(
        parameters.maximum_segments,
        max(1, int(duration_seconds // parameters.minimum_segment_seconds)),
        count,
    )
    prefix = np.vstack(
        [np.zeros((1, matrix.shape[1]), dtype=np.float32), np.cumsum(matrix, axis=0)]
    )
    costs = np.full((maximum + 1, count + 1), np.inf, dtype=np.float64)
    previous = np.full((maximum + 1, count + 1), -1, dtype=int)
    costs[0, 0] = 0.0
    for segment_count in range(1, maximum + 1):
        for end in range(1, count + 1):
            for start in range(segment_count - 1, end):
                if not np.isfinite(costs[segment_count - 1, start]):
                    continue
                duration = boundaries[end] - boundaries[start]
                if duration + 1e-6 < parameters.minimum_segment_seconds and not (
                    start == 0
                    and end == count
                    and duration_seconds < parameters.minimum_segment_seconds
                ):
                    continue
                value = costs[segment_count - 1, start] + _segment_cost(prefix, start, end)
                value += parameters.segment_penalty if segment_count > 1 else 0.0
                if value < costs[segment_count, end]:
                    costs[segment_count, end] = value
                    previous[segment_count, end] = start
    chosen_count = int(np.argmin(costs[1:, count])) + 1
    if not np.isfinite(costs[chosen_count, count]):
        chosen_count = 1
    ranges: list[tuple[int, int]] = []
    end = count
    for segment_count in range(chosen_count, 0, -1):
        start = int(previous[segment_count, end])
        if start < 0:
            start = 0
        ranges.append((start, end))
        end = start
    ranges.reverse()

    result: list[TemporalSegment] = []
    for segment_index, (start, end) in enumerate(ranges):
        centroid = _centroid(matrix[start:end])
        similarities = matrix[start:end] @ centroid
        representative = start + int(np.argmax(similarities))
        result.append(
            TemporalSegment(
                segment_index,
                float(boundaries[start]),
                float(boundaries[end]),
                centroid,
                float(np.mean(similarities)),
                windows[representative].index,
            )
        )
    return tuple(result)


def _spherical_kmeans(
    matrix: np.ndarray, clusters: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    centers = [matrix[int(rng.integers(len(matrix)))]]
    while len(centers) < clusters:
        similarity = matrix @ np.stack(centers).T
        distance = 1 - similarity.max(axis=1)
        centers.append(matrix[int(np.argmax(distance))])
    center_matrix = np.stack(centers)
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(50):
        updated_labels = np.argmax(matrix @ center_matrix.T, axis=1)
        updated_centers = []
        for cluster in range(clusters):
            members = matrix[updated_labels == cluster]
            if not len(members):
                updated_centers.append(matrix[int(rng.integers(len(matrix)))])
            else:
                updated_centers.append(_centroid(members))
        updated_matrix = np.stack(updated_centers)
        if np.array_equal(updated_labels, labels) and np.allclose(updated_matrix, center_matrix):
            labels, center_matrix = updated_labels, updated_matrix
            break
        labels, center_matrix = updated_labels, updated_matrix
    dispersion = float(np.mean(1 - np.sum(matrix * center_matrix[labels], axis=1)))
    return labels, center_matrix, dispersion


def _mode_solution(
    windows: Sequence[WindowVector],
    duration_seconds: float,
    parameters: AudioProfileParameters,
) -> tuple[np.ndarray, np.ndarray]:
    selected_indices = _decorrelated_indices(windows)
    selected = np.stack([windows[index].vector for index in selected_indices])
    maximum = min(parameters.maximum_modes, len(selected))
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for clusters in range(1, maximum + 1):
        runs = [
            _spherical_kmeans(selected, clusters, 7919 + seed)
            for seed in range(parameters.clustering_seeds)
        ]
        labels, centers, dispersion = min(runs, key=lambda item: item[2])
        comparisons = [
            adjusted_rand_score(runs[left][0], runs[right][0])
            for left in range(len(runs))
            for right in range(left + 1, len(runs))
        ]
        stability = float(np.mean(comparisons)) if comparisons else 1.0
        selected_durations = np.asarray(
            [windows[index].duration_seconds for index in selected_indices]
        )
        supports = np.asarray(
            [selected_durations[labels == cluster].sum() for cluster in range(clusters)]
        )
        minimum_support = max(
            parameters.minimum_mode_seconds,
            duration_seconds * parameters.minimum_mode_share,
        )
        valid = bool(np.all(supports + 1e-6 >= minimum_support))
        if clusters > 1 and stability < parameters.minimum_mode_stability:
            valid = False
        if clusters > 1:
            similarities = centers @ centers.T
            if np.any(
                similarities[np.triu_indices(clusters, 1)] >= parameters.mode_merge_similarity
            ):
                valid = False
        if valid or clusters == 1:
            score = dispersion + parameters.mode_penalty * (clusters - 1)
            candidates.append((score, clusters, labels, centers))
    _, _, _, centers = min(candidates, key=lambda item: (item[0], item[1]))
    all_matrix = np.stack([window.vector for window in windows])
    all_labels = np.argmax(all_matrix @ centers.T, axis=1)
    return all_labels, centers


def _musical_modes(
    windows: Sequence[WindowVector],
    duration_seconds: float,
    parameters: AudioProfileParameters,
) -> tuple[MusicalMode, ...]:
    labels, centers = _mode_solution(windows, duration_seconds, parameters)
    matrix = np.stack([window.vector for window in windows])
    boundaries = _interval_boundaries(windows, duration_seconds)
    modes: list[MusicalMode] = []
    for mode_index, center in enumerate(centers):
        member_indices = np.flatnonzero(labels == mode_index)
        similarities = matrix[member_indices] @ center
        representative = int(member_indices[int(np.argmax(similarities))])
        intervals: list[ModeInterval] = []
        run_start: int | None = None
        for index in range(len(windows) + 1):
            belongs = index < len(windows) and labels[index] == mode_index
            if belongs and run_start is None:
                run_start = index
            if not belongs and run_start is not None:
                intervals.append(
                    ModeInterval(float(boundaries[run_start]), float(boundaries[index]))
                )
                run_start = None
        support = sum(interval.end_seconds - interval.start_seconds for interval in intervals)
        modes.append(
            MusicalMode(
                mode_index,
                center,
                support / max(duration_seconds, 1e-8),
                float(np.mean(similarities)),
                windows[representative].index,
                tuple(intervals),
            )
        )
    modes.sort(key=lambda mode: (-mode.duration_weight, mode.index))
    remapped = []
    for index, mode in enumerate(modes):
        remapped.append(
            MusicalMode(
                index,
                mode.vector,
                mode.duration_weight,
                mode.cohesion,
                mode.representative_window_index,
                mode.intervals,
            )
        )
    return tuple(remapped)


def derive_audio_profile(
    windows: Sequence[WindowVector],
    duration_seconds: float,
    timestamps_exact: bool = True,
    parameters: AudioProfileParameters = DEFAULT_PARAMETERS,
) -> TrackAudioProfile:
    if not windows:
        raise ValueError("An audio profile requires at least one window embedding")
    if duration_seconds <= 0:
        raise ValueError("An audio profile requires a positive track duration")
    ordered = sorted(windows, key=lambda window: (window.start_seconds, window.index))
    normalized_windows = [
        WindowVector(item.index, item.start_seconds, item.end_seconds, _normalize(item.vector))
        for item in ordered
    ]
    matrix = np.stack([item.vector for item in normalized_windows])
    raw_mean = matrix.mean(axis=0)
    resultant_length = float(np.linalg.norm(raw_mean))
    global_overlap = _normalize(raw_mean)
    decorrelated = matrix[_decorrelated_indices(normalized_windows)]
    global_decorrelated = _centroid(decorrelated)
    global_similarities = matrix @ global_overlap
    adjacent_changes = 1 - np.sum(matrix[:-1] * matrix[1:], axis=1)
    if not len(adjacent_changes):
        adjacent_changes = np.asarray([0.0], dtype=np.float32)
    return TrackAudioProfile(
        global_overlap=global_overlap,
        global_decorrelated=global_decorrelated,
        resultant_length=resultant_length,
        mean_global_similarity=float(np.mean(global_similarities)),
        p05_global_similarity=float(np.percentile(global_similarities, 5)),
        adjacent_change_mean=float(np.mean(adjacent_changes)),
        adjacent_change_p95=float(np.percentile(adjacent_changes, 95)),
        timestamps_exact=timestamps_exact,
        segments=_temporal_segments(normalized_windows, duration_seconds, parameters),
        modes=_musical_modes(normalized_windows, duration_seconds, parameters),
    )


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _create_profile_run(
    connection: psycopg.Connection,
    model_name: str,
    parameters: AudioProfileParameters,
) -> uuid.UUID:
    if model_name not in SUPPORTED_PROFILE_MODELS:
        raise ValueError(f"Unsupported audio-profile source model: {model_name}")
    config = profile_config(model_name, parameters)
    environment = {"python": platform.python_version(), "numpy": np.__version__}
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO analysis_runs
                 (kind, model_name, model_revision, config_hash, config, environment,
                  device, precision, status, started_at)
               VALUES ('audio_profile',%s,%s,%s,%s,%s,'cpu','float32','running',now())
               ON CONFLICT (kind, model_name, model_revision, config_hash)
               DO UPDATE SET id=analysis_runs.id
               RETURNING id""",
            (
                model_name,
                AUDIO_PROFILE_REVISION,
                config_hash,
                Jsonb(config),
                Jsonb(environment),
            ),
        )
        return cursor.fetchone()["id"]


def _store_profile(
    connection: psycopg.Connection,
    track_id: uuid.UUID,
    source_run_id: uuid.UUID,
    profile_run_id: uuid.UUID,
    model_name: str,
    profile: TrackAudioProfile,
) -> None:
    dimension = len(profile.global_overlap)
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO track_audio_profiles
                 (track_id, profile_run_id, source_run_id, model_name, dimension,
                  global_overlap_embedding, global_decorrelated_embedding,
                  resultant_length, mean_global_similarity, p05_global_similarity,
                  adjacent_change_mean, adjacent_change_p95, timestamps_exact, mode_count)
               VALUES (%s,%s,%s,%s,%s,%s::vector,%s::vector,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (track_id, profile_run_id, source_run_id) DO UPDATE SET
                 dimension=EXCLUDED.dimension,
                 model_name=EXCLUDED.model_name,
                 global_overlap_embedding=EXCLUDED.global_overlap_embedding,
                 global_decorrelated_embedding=EXCLUDED.global_decorrelated_embedding,
                 resultant_length=EXCLUDED.resultant_length,
                 mean_global_similarity=EXCLUDED.mean_global_similarity,
                 p05_global_similarity=EXCLUDED.p05_global_similarity,
                 adjacent_change_mean=EXCLUDED.adjacent_change_mean,
                 adjacent_change_p95=EXCLUDED.adjacent_change_p95,
                 timestamps_exact=EXCLUDED.timestamps_exact,
                 mode_count=EXCLUDED.mode_count,
                 created_at=now()
               RETURNING id""",
            (
                track_id,
                profile_run_id,
                source_run_id,
                model_name,
                dimension,
                _vector_literal(profile.global_overlap),
                _vector_literal(profile.global_decorrelated),
                profile.resultant_length,
                profile.mean_global_similarity,
                profile.p05_global_similarity,
                profile.adjacent_change_mean,
                profile.adjacent_change_p95,
                profile.timestamps_exact,
                len(profile.modes),
            ),
        )
        profile_id = cursor.fetchone()["id"]
        cursor.execute(
            "DELETE FROM audio_temporal_segments WHERE profile_id=%s",
            (profile_id,),
        )
        cursor.execute(
            "DELETE FROM audio_modes WHERE profile_id=%s",
            (profile_id,),
        )
        cursor.executemany(
            """INSERT INTO audio_temporal_segments
                 (profile_id, segment_index, start_seconds, end_seconds,
                  dimension, embedding, cohesion, representative_window_index)
               VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s)""",
            [
                (
                    profile_id,
                    item.index,
                    item.start_seconds,
                    item.end_seconds,
                    len(item.vector),
                    _vector_literal(item.vector),
                    item.cohesion,
                    item.representative_window_index,
                )
                for item in profile.segments
            ],
        )
        for item in profile.modes:
            cursor.execute(
                """INSERT INTO audio_modes
                     (profile_id, mode_index, dimension, embedding,
                      duration_weight, cohesion, representative_window_index)
                   VALUES (%s,%s,%s,%s::vector,%s,%s,%s) RETURNING id""",
                (
                    profile_id,
                    item.index,
                    len(item.vector),
                    _vector_literal(item.vector),
                    item.duration_weight,
                    item.cohesion,
                    item.representative_window_index,
                ),
            )
            mode_id = cursor.fetchone()["id"]
            cursor.executemany(
                """INSERT INTO audio_mode_intervals (mode_id, interval_index, start_seconds, end_seconds)
                   VALUES (%s,%s,%s,%s)""",
                [
                    (mode_id, index, interval.start_seconds, interval.end_seconds)
                    for index, interval in enumerate(item.intervals)
                ],
            )


def build_audio_profiles(
    track_ids: Sequence[uuid.UUID] | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
    parameters: AudioProfileParameters = DEFAULT_PARAMETERS,
    model_name: str = "muq_mulan",
) -> dict[str, object]:
    from core.embeddings.representations import configure_representations

    report = progress or (lambda _: None)
    summary: dict[str, object] = {
        "model": model_name,
        "total": 0,
        "profiled": 0,
        "failed": 0,
        "profile_run_id": None,
    }
    with psycopg.connect(require_database_url(), row_factory=dict_row) as connection:
        configure_representations(connection)
        profile_run_id = _create_profile_run(connection, model_name, parameters)
        summary["profile_run_id"] = str(profile_run_id)
        connection.commit()
        restriction = " AND e.track_id=ANY(%s)" if track_ids is not None else ""
        arguments = [list(track_ids)] if track_ids is not None else []
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT DISTINCT ON (e.track_id)
                          e.track_id, e.run_id AS source_run_id, t.duration_seconds,
                          ar.config->>'window_seconds' AS window_seconds,
                          ar.config->>'stride_seconds' AS stride_seconds
                    FROM current_embeddings e
                    JOIN analysis_runs ar ON ar.id=e.run_id
                    JOIN tracks t ON t.id=e.track_id
                    WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
                      AND ar.model_name=%s
                      AND NOT EXISTS (
                        SELECT 1 FROM track_audio_profiles tap
                        JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                        WHERE tap.track_id=e.track_id
                          AND tap.source_run_id=e.run_id
                          AND tap.model_name=%s
                          AND profile_run.kind='audio_profile'
                          AND profile_run.id=%s
                      ){restriction}
                    ORDER BY e.track_id, ar.created_at DESC""",
                [model_name, model_name, profile_run_id, *arguments],
            )
            tracks = cursor.fetchall()
        summary["total"] = len(tracks)
        for completed, track in enumerate(tracks):
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """SELECT window_index, window_start_seconds, window_end_seconds,
                                  embedding::text AS embedding
                           FROM embeddings
                           WHERE track_id=%s AND run_id=%s AND embedding_type='audio-window'
                           ORDER BY window_index""",
                        (track["track_id"], track["source_run_id"]),
                    )
                    rows = cursor.fetchall()
                if not rows:
                    raise ValueError(f"{model_name} run has no stored window embeddings")
                duration = float(track["duration_seconds"])
                exact = all(
                    row["window_start_seconds"] is not None
                    and row["window_end_seconds"] is not None
                    for row in rows
                )
                window_seconds = float(track["window_seconds"] or 10)
                stride_seconds = float(track["stride_seconds"] or 5)
                if exact:
                    duration = max(
                        duration,
                        max(float(row["window_end_seconds"]) for row in rows),
                    )
                elif len(rows) > 1:
                    # With the ingestion rule, the penultimate window is still
                    # stride-aligned even when the final window has an irregular start.
                    minimum_compatible_duration = window_seconds + stride_seconds * max(
                        0, len(rows) - 2
                    )
                    duration = max(duration, minimum_compatible_duration)
                elif duration <= 0:
                    duration = window_seconds
                inferred = infer_window_ranges(
                    len(rows),
                    duration,
                    window_seconds,
                    stride_seconds,
                )
                windows = [
                    WindowVector(
                        int(row["window_index"]),
                        float(row["window_start_seconds"] if exact else inferred[index][0]),
                        float(row["window_end_seconds"] if exact else inferred[index][1]),
                        np.fromstring(row["embedding"].strip("[]"), sep=","),
                    )
                    for index, row in enumerate(rows)
                ]
                profile = derive_audio_profile(windows, duration, exact, parameters)
                _store_profile(
                    connection,
                    track["track_id"],
                    track["source_run_id"],
                    profile_run_id,
                    model_name,
                    profile,
                )
                connection.commit()
                summary["profiled"] = int(summary["profiled"]) + 1
            except Exception:
                connection.rollback()
                summary["failed"] = int(summary["failed"]) + 1
                logger.exception("Failed to derive audio profile for track %s", track["track_id"])
            report(
                {
                    "phase": "audio_profiles",
                    "message": f"Deriving {model_name} multi-vector audio profiles",
                    "completed": completed + 1,
                    "total": len(tracks),
                    "unit": "tracks",
                    "summary": summary,
                }
            )
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE analysis_runs SET status='complete', finished_at=now() WHERE id=%s",
                (profile_run_id,),
            )
        connection.commit()
    return summary
