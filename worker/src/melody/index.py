from __future__ import annotations

from core.settings import get_settings

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import uuid

# The service package is read-only for its unprivileged runtime user. Numba must
# compile librosa's pitch helpers into a writable cache directory.
os.environ.setdefault("NUMBA_CACHE_DIR", "/models/torch/numba")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import librosa
from numba import njit, prange
import numpy as np
import psycopg
from psycopg.rows import dict_row, tuple_row
from psycopg.types.json import Jsonb

from core.media.audio import decode_audio
from core.revisions import (
    MELODY_CONTOUR_REVISION,
    MELODY_EXTRACTOR,
    SEPARATION_REVISION as MELODY_SEPARATOR_REVISION,
    SEPARATOR_MODEL_ID as MELODY_SEPARATOR_MODEL,
)
from worker.media.preprocessing import melody_waveforms, get_check, prepare_audio
from core.integrations.navidrome import NavidromeClient

CATALOG_SAMPLE_RATE = 44_100
QUERY_SAMPLE_RATE = 24_000
CONTOUR_HZ = 10
DEFAULT_CORPUS_SIZE = 50
_TEMPO_RATIOS = (0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5)
# Contours only change while a sync or corpus build is running. Caching the
# parsed arrays removes the per-query array decoding cost.
_CONTOUR_CACHE: dict[str, object] = {"key": None, "contours": ()}
_CACHE_LOCK = threading.Lock()


def _smooth_pitch(pitch: np.ndarray, voiced: np.ndarray, radius: int = 2) -> np.ndarray:
    result = pitch.astype(np.float32, copy=True)
    for index in np.flatnonzero(voiced):
        left, right = max(0, index - radius), min(len(pitch), index + radius + 1)
        values = pitch[left:right][voiced[left:right]]
        if values.size:
            result[index] = np.median(values)
    return result


def _fill_short_gaps(
    pitch: np.ndarray, voiced: np.ndarray, maximum: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    pitch, voiced = pitch.copy(), voiced.copy()
    index = 0
    while index < len(voiced):
        if voiced[index]:
            index += 1
            continue
        end = index
        while end < len(voiced) and not voiced[end]:
            end += 1
        if index > 0 and end < len(voiced) and end - index <= maximum:
            pitch[index:end] = np.linspace(pitch[index - 1], pitch[end], end - index + 2)[1:-1]
            voiced[index:end] = True
        index = end
    return pitch, voiced


def _to_contour(
    pitch_hz: np.ndarray, voiced: np.ndarray, source_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, round(source_hz / CONTOUR_HZ))
    bins = int(np.ceil(len(pitch_hz) / step))
    pitch = np.zeros(bins, dtype=np.float32)
    mask = np.zeros(bins, dtype=bool)
    for index in range(bins):
        left, right = index * step, min(len(pitch_hz), (index + 1) * step)
        values = pitch_hz[left:right][
            voiced[left:right] & np.isfinite(pitch_hz[left:right]) & (pitch_hz[left:right] > 0)
        ]
        if values.size:
            pitch[index] = float(np.median(69 + 12 * np.log2(values / 440.0)))
            mask[index] = True
    pitch, mask = _fill_short_gaps(pitch, mask)
    return _smooth_pitch(pitch, mask), mask


def extract_waveform_contour(
    waveform: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    from essentia.standard import EqualLoudness, PredominantPitchMelodia

    if sample_rate != CATALOG_SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=CATALOG_SAMPLE_RATE)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    equalized = EqualLoudness(sampleRate=CATALOG_SAMPLE_RATE)(waveform)
    hop = 256
    pitch, confidence = PredominantPitchMelodia(
        sampleRate=CATALOG_SAMPLE_RATE,
        frameSize=2048,
        hopSize=hop,
        minFrequency=70,
        maxFrequency=1600,
    )(equalized)
    pitch = np.asarray(pitch, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    voiced = (pitch > 0) & np.isfinite(pitch) & (confidence > 0)
    return _to_contour(pitch, voiced, CATALOG_SAMPLE_RATE / hop)


def extract_catalog_contour(audio: bytes) -> tuple[np.ndarray, np.ndarray]:
    return extract_waveform_contour(decode_audio(audio, CATALOG_SAMPLE_RATE), CATALOG_SAMPLE_RATE)


def separate_melody_sources(audio: bytes) -> dict[str, tuple[np.ndarray, int]]:
    vocals, accompaniment = melody_waveforms(audio, check=get_check())
    return {
        "vocals": (vocals, CATALOG_SAMPLE_RATE),
        "accompaniment": (accompaniment, CATALOG_SAMPLE_RATE),
    }


def extract_hum_contour(audio: bytes) -> tuple[np.ndarray, np.ndarray]:
    waveform = decode_audio(audio, QUERY_SAMPLE_RATE)
    frame_length, hop = 2048, 240
    f0, voiced, probability = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=QUERY_SAMPLE_RATE,
        frame_length=frame_length,
        hop_length=hop,
    )
    valid = voiced & np.isfinite(f0) & (probability >= 0.55)
    pitch, mask = _to_contour(np.nan_to_num(f0), valid, QUERY_SAMPLE_RATE / hop)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("No stable hummed pitch was detected")
    pitch, mask = pitch[indices[0] : indices[-1] + 1], mask[indices[0] : indices[-1] + 1]
    if mask.sum() < 25:
        raise ValueError("Hum a clear melody for at least three seconds")
    return pitch, mask


def _relative(pitch: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    result = pitch.astype(np.float32, copy=True)
    if voiced.any():
        result[voiced] -= np.median(result[voiced])
    return result


def _resample(values: np.ndarray, mask: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    source = np.arange(len(values), dtype=np.float32)
    target = np.linspace(0, max(0, len(values) - 1), length)
    filled = values.copy()
    known = np.flatnonzero(mask)
    if known.size < 2:
        return np.zeros(length, dtype=np.float32), np.zeros(length, dtype=bool)
    filled = np.interp(source, known, values[known])
    output = np.interp(target, source, filled).astype(np.float32)
    output_mask = np.interp(target, source, mask.astype(np.float32)) >= 0.5
    return output, output_mask


def _dtw_cost(
    query: np.ndarray, query_mask: np.ndarray, target: np.ndarray, target_mask: np.ndarray
) -> float:
    query, target = _relative(query, query_mask), _relative(target, target_mask)
    both_voiced = query_mask[:, None] & target_mask[None, :]
    both_unvoiced = ~query_mask[:, None] & ~target_mask[None, :]
    delta = np.minimum(np.abs(query[:, None] - target[None, :]), 6.0) / 3.0
    local = np.where(both_voiced, delta, np.where(both_unvoiced, 0.2, 1.0)).astype(np.float32)
    accumulated = librosa.sequence.dtw(C=local, backtrack=False)
    return float(accumulated[-1, -1] / max(len(query), len(target)))


@njit(cache=True, inline="always")
def _median_in_place(values: np.ndarray, count: int) -> float:
    """Return the median of the populated prefix using three-way quickselect."""
    middle = count // 2
    left, right = 0, count - 1
    while left < right:
        pivot = values[(left + right) // 2]
        lower, index, upper = left, left, right
        while index <= upper:
            if values[index] < pivot:
                values[lower], values[index] = values[index], values[lower]
                lower += 1
                index += 1
            elif values[index] > pivot:
                values[index], values[upper] = values[upper], values[index]
                upper -= 1
            else:
                index += 1
        if middle < lower:
            right = lower - 1
        elif middle > upper:
            left = upper + 1
        else:
            break
    upper_median = values[middle]
    if count % 2:
        return upper_median
    lower_median = values[0]
    for index in range(1, middle):
        if values[index] > lower_median:
            lower_median = values[index]
    return (lower_median + upper_median) / 2.0


@njit(cache=True, nogil=True)
def _coarse_tempo(
    query_values: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    voiced_offsets: np.ndarray,
    width: int,
    stride: int,
    minimum_voiced: float,
    query_length: int,
) -> tuple[float, int]:
    """Find the best window for one tempo without allocating a matrix.

    query_values and voiced_offsets cover only the voiced query positions, so
    unvoiced frames cost no iterations. A window is abandoned once its maximum
    possible voiced overlap can no longer reach minimum_voiced, which rejects
    exactly the windows the full count would reject.
    """
    voiced_total = len(query_values)
    target_values = np.empty(voiced_total, dtype=np.float32)
    query_pair_values = np.empty(voiced_total, dtype=np.float32)
    # _median_in_place partitions its input, so each start medians copies while
    # the delta loop keeps reading the unpermuted values.
    query_scratch = np.empty(voiced_total, dtype=np.float32)
    target_scratch = np.empty(voiced_total, dtype=np.float32)
    best_score = np.inf
    best_start = -1

    for start in range(0, len(target) - width + 1, stride):
        voiced_count = 0
        feasible = True
        for index in range(voiced_total):
            target_index = start + voiced_offsets[index]
            if target_mask[target_index]:
                query_pair_values[voiced_count] = query_values[index]
                target_values[voiced_count] = target[target_index]
                voiced_count += 1
            elif voiced_count + (voiced_total - index - 1) < minimum_voiced:
                feasible = False
                break
        if not feasible or voiced_count < minimum_voiced:
            continue

        query_scratch[:voiced_count] = query_pair_values[:voiced_count]
        target_scratch[:voiced_count] = target_values[:voiced_count]
        query_median = _median_in_place(query_scratch, voiced_count)
        target_median = _median_in_place(target_scratch, voiced_count)

        delta_sum = 0.0
        for index in range(voiced_count):
            delta = abs(
                (query_pair_values[index] - query_median) - (target_values[index] - target_median)
            )
            delta_sum += min(delta, 6.0)
        score = delta_sum / voiced_count + (1.0 - voiced_count / query_length)
        if score < best_score:
            best_score = score
            best_start = start

    return best_score, best_start


def _coarse_match(
    query: np.ndarray, query_mask: np.ndarray, target: np.ndarray, target_mask: np.ndarray
) -> tuple[float, int, np.ndarray, np.ndarray] | None:
    query_length = len(query)
    stride = max(1, CONTOUR_HZ // 2)
    minimum_voiced = float(query_mask.sum()) * 0.55
    voiced_indices = np.flatnonzero(query_mask)
    query_values = query[voiced_indices]
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    for tempo in _TEMPO_RATIOS:
        width = max(20, round(query_length * tempo))
        if width > len(target):
            continue
        offsets = np.rint(np.linspace(0, width - 1, query_length)).astype(np.int64)
        score, start = _coarse_tempo(
            query_values,
            target,
            target_mask,
            offsets[voiced_indices],
            width,
            stride,
            minimum_voiced,
            query_length,
        )
        if start >= 0 and (best is None or score < best[0]):
            indices = start + offsets
            best = (float(score), int(start), target[indices], target_mask[indices])
    return best


@njit(cache=True, nogil=True, parallel=True)
def _coarse_motif_batch(
    query_values_padded: np.ndarray,
    voiced_counts: np.ndarray,
    query_lengths: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    widths: np.ndarray,
    voiced_offsets: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan every prepared motif and tempo for one catalog contour.

    Motifs are independent, so scanning them in parallel is bit-exact with the
    serial loop: every start position is scored with identical arithmetic and
    each motif keeps its own deterministic minimum.
    """
    motif_count, tempo_count = widths.shape
    best_scores = np.full(motif_count, np.inf)
    best_starts = np.full(motif_count, -1, dtype=np.int64)
    best_tempos = np.full(motif_count, -1, dtype=np.int64)
    for motif_index in prange(motif_count):
        length = query_lengths[motif_index]
        voiced_count = voiced_counts[motif_index]
        query_values = query_values_padded[motif_index, :voiced_count]
        minimum_voiced = float(voiced_count) * 0.55
        for tempo_index in range(tempo_count):
            width = widths[motif_index, tempo_index]
            if width > len(target):
                continue
            score, start = _coarse_tempo(
                query_values,
                target,
                target_mask,
                voiced_offsets[motif_index, tempo_index, :voiced_count],
                width,
                stride,
                minimum_voiced,
                length,
            )
            if start >= 0 and score < best_scores[motif_index]:
                best_scores[motif_index] = score
                best_starts[motif_index] = start
                best_tempos[motif_index] = tempo_index
    return best_scores, best_starts, best_tempos


def _prepare_motifs(
    windows: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    lengths = np.asarray([len(query) for query, _ in windows], dtype=np.int64)
    voiced_counts = np.asarray([int(mask.sum()) for _, mask in windows], dtype=np.int64)
    maximum = int(lengths.max())
    maximum_voiced = int(voiced_counts.max())
    # Only voiced query positions are scored, so pack their values and their
    # per-tempo target offsets once instead of re-deriving them per window.
    query_values = np.zeros((len(windows), maximum_voiced), dtype=np.float32)
    queries = np.zeros((len(windows), maximum), dtype=np.float32)
    masks = np.zeros((len(windows), maximum), dtype=bool)
    voiced_offsets = np.zeros((len(windows), len(_TEMPO_RATIOS), maximum_voiced), dtype=np.int64)
    widths = np.zeros((len(windows), len(_TEMPO_RATIOS)), dtype=np.int64)
    offsets = np.zeros((len(windows), len(_TEMPO_RATIOS), maximum), dtype=np.int64)
    for motif_index, (query, mask) in enumerate(windows):
        length = len(query)
        queries[motif_index, :length] = query
        masks[motif_index, :length] = mask
        voiced_indices = np.flatnonzero(mask)
        query_values[motif_index, : len(voiced_indices)] = query[voiced_indices]
        length = len(query)
        for tempo_index, tempo in enumerate(_TEMPO_RATIOS):
            width = max(20, round(length * tempo))
            widths[motif_index, tempo_index] = width
            motif_offsets = np.rint(np.linspace(0, width - 1, length)).astype(np.int64)
            offsets[motif_index, tempo_index, :length] = motif_offsets
            voiced_offsets[motif_index, tempo_index, : len(voiced_indices)] = motif_offsets[
                voiced_indices
            ]
    return query_values, voiced_counts, lengths, queries, masks, widths, voiced_offsets, offsets


def _match_prepared_motifs(
    prepared: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    target: np.ndarray,
    target_mask: np.ndarray,
) -> list[tuple[float, float]]:
    query_values, voiced_counts, lengths, queries, masks, widths, voiced_offsets, offsets = prepared
    _, starts, tempos = _coarse_motif_batch(
        query_values,
        voiced_counts,
        lengths,
        target,
        target_mask,
        widths,
        voiced_offsets,
        max(1, CONTOUR_HZ // 2),
    )
    matches: list[tuple[float, float]] = []
    for motif_index, start in enumerate(starts):
        tempo_index = tempos[motif_index]
        if start < 0 or tempo_index < 0:
            matches.append((float("inf"), 0.0))
            continue
        length = lengths[motif_index]
        indices = start + offsets[motif_index, tempo_index, :length]
        cost = _dtw_cost(
            queries[motif_index, :length],
            masks[motif_index, :length],
            target[indices],
            target_mask[indices],
        )
        matches.append((cost, float(start) / CONTOUR_HZ))
    return matches


def match_contour(
    query: np.ndarray, query_mask: np.ndarray, target: np.ndarray, target_mask: np.ndarray
) -> tuple[float, float]:
    """Return a lower-is-better cost and target offset in seconds."""
    best = _coarse_match(query, query_mask, target, target_mask)
    if best is None:
        return float("inf"), 0.0
    return _dtw_cost(query, query_mask, best[2], best[3]), best[1] / CONTOUR_HZ


def _create_run(connection: psycopg.Connection, corpus_id: uuid.UUID) -> uuid.UUID:
    config = {
        "purpose": "hum_search",
        "corpus_id": str(corpus_id),
        "extractor": MELODY_EXTRACTOR,
        "separator_model": MELODY_SEPARATOR_MODEL,
        "separator_revision": MELODY_SEPARATOR_REVISION,
        "accompaniment": "stereo-mix-minus-vocals-then-mean-v1",
        "sources": ["full-mix", "vocals", "accompaniment"],
        "contour_hz": CONTOUR_HZ,
        "matcher": "relative-pitch-subsequence-dtw-v1",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO analysis_runs (kind,model_name,model_revision,config_hash,config,environment,device,precision,status,started_at)
               VALUES ('hum_corpus','melody_contour',%s,%s,%s,%s,'mixed','float32','running',now()) RETURNING id""",
            (
                MELODY_CONTOUR_REVISION,
                config_hash,
                Jsonb(config),
                Jsonb({"python": platform.python_version()}),
            ),
        )
        return cursor.fetchone()["id"]


def create_sync_run(connection: psycopg.Connection) -> uuid.UUID:
    config = {
        "extractor": MELODY_EXTRACTOR,
        "separator_model": MELODY_SEPARATOR_MODEL,
        "separator_revision": MELODY_SEPARATOR_REVISION,
        "accompaniment": "stereo-mix-minus-vocals-then-mean-v1",
        "sources": ["full-mix", "vocals", "accompaniment"],
        "contour_hz": CONTOUR_HZ,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO analysis_runs (kind,model_name,model_revision,config_hash,config,environment,device,precision,status,started_at)
               VALUES ('melody_contour','melody_contour',%s,%s,%s,%s,'mixed','float32','running',now())
               ON CONFLICT (kind,model_name,model_revision,config_hash)
               DO UPDATE SET status='running',started_at=now(),finished_at=NULL RETURNING id""",
            (
                MELODY_CONTOUR_REVISION,
                config_hash,
                Jsonb(config),
                Jsonb({"python": platform.python_version()}),
            ),
        )
        row = cursor.fetchone()
        return row["id"] if isinstance(row, dict) else row[0]


def track_contours(audio: bytes) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Usable melody contours per source. The same code runs locally and on Modal."""
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {"full-mix": extract_catalog_contour(audio)}
    # A preparation failure must not mark a full-mix-only result as a completed
    # Roformer recipe. The enclosing track job handles retry/failure accounting.
    for source, (waveform, sample_rate) in separate_melody_sources(audio).items():
        get_check()()
        sources[source] = extract_waveform_contour(waveform, sample_rate)
    usable = {source: contour for source, contour in sources.items() if contour[1].sum() >= 30}
    if not usable:
        raise ValueError("No usable melody source")
    return usable


def store_track_contours(
    connection: psycopg.Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    audio: bytes,
    usable: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> int:
    """Store a track's contours, computed here, on Modal for this batch, or passed in."""
    if usable is None:
        from worker.remote.session import current as current_remote

        remote = current_remote()
        usable = remote.melody_one(audio) if remote else track_contours(audio)
    with connection.cursor() as cursor:
        for source, (pitch, voiced) in usable.items():
            cursor.execute(
                """INSERT INTO melody_contours (track_id,run_id,source,hop_seconds,pitch,voiced)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (track_id,run_id,source) DO UPDATE
                   SET hop_seconds=EXCLUDED.hop_seconds,pitch=EXCLUDED.pitch,voiced=EXCLUDED.voiced,
                       created_at=clock_timestamp()""",
                (track_id, run_id, source, 1 / CONTOUR_HZ, pitch.tolist(), voiced.tolist()),
            )
    return len(usable)


def build_corpus(
    corpus_id: uuid.UUID,
    user_id: uuid.UUID,
    credentials: tuple[str, str, str],
    track_limit: int = DEFAULT_CORPUS_SIZE,
    progress: Callable[[dict[str, object]], None] | None = None,
    track_ids: set[uuid.UUID] | None = None,
) -> dict[str, int]:
    report = progress or (lambda _: None)
    completed = failed = contours_stored = 0
    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as connection:
        run_id = _create_run(connection, corpus_id)
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE hum_corpora SET run_id=%s,status='building' WHERE id=%s AND user_id=%s",
                (run_id, corpus_id, user_id),
            )
            cursor.execute(
                """SELECT DISTINCT ON (utl.track_id) utl.track_id,utl.external_id,t.title FROM user_track_links utl JOIN tracks t ON t.id=utl.track_id WHERE utl.user_id=%s ORDER BY utl.track_id""",
                (user_id,),
            )
            candidates = cursor.fetchall()
        if track_ids is not None:
            tracks = [track for track in candidates if track["track_id"] in track_ids][:track_limit]
        else:
            np.random.default_rng().shuffle(candidates)
            tracks = candidates[:track_limit]
        connection.commit()
        with NavidromeClient(*credentials) as client:
            for index, track in enumerate(tracks):
                report(
                    {
                        "phase": "melody-index",
                        "completed": index,
                        "total": len(tracks),
                        "message": f"Separating melody sources for {track['title']}",
                    }
                )
                try:
                    report(
                        {
                            "phase": "preprocess",
                            "completed": index,
                            "total": len(tracks),
                            "message": f"Preparing shared melody audio for {track['title']}",
                        }
                    )
                    audio = client.audio_bytes(track["external_id"])
                    prepare_audio(
                        audio, mono_rates=(44100,), stereo=True, melody=True, check=get_check()
                    )
                    report(
                        {
                            "phase": "melody-index",
                            "message": f"Extracting melody from {track['title']}",
                        }
                    )
                    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {
                        "full-mix": extract_catalog_contour(audio),
                    }
                    for source, (waveform, sample_rate) in separate_melody_sources(audio).items():
                        get_check()()
                        sources[source] = extract_waveform_contour(waveform, sample_rate)
                    usable = {
                        source: contour
                        for source, contour in sources.items()
                        if contour[1].sum() >= 30
                    }
                    if not usable:
                        raise ValueError("No usable melody source")
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO hum_corpus_tracks (corpus_id,track_id) VALUES (%s,%s)",
                            (corpus_id, track["track_id"]),
                        )
                        for source, (pitch, voiced) in usable.items():
                            cursor.execute(
                                """INSERT INTO melody_contours (track_id,run_id,source,hop_seconds,pitch,voiced) VALUES (%s,%s,%s,%s,%s,%s)""",
                                (
                                    track["track_id"],
                                    run_id,
                                    source,
                                    1 / CONTOUR_HZ,
                                    pitch.tolist(),
                                    voiced.tolist(),
                                ),
                            )
                    connection.commit()
                    completed += 1
                    contours_stored += len(usable)
                except Exception:
                    connection.rollback()
                    failed += 1
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE analysis_runs SET status='complete',finished_at=now() WHERE id=%s",
                (run_id,),
            )
            cursor.execute(
                "UPDATE hum_corpora SET status='complete',completed_at=now() WHERE id=%s",
                (corpus_id,),
            )
        connection.commit()
    return {"tracks": completed, "failed": failed, "contours": contours_stored}


def _contour_cache_key(connection: psycopg.Connection, user_id: uuid.UUID) -> tuple[str, int, str]:
    with connection.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(
            """SELECT count(*),coalesce(max(mc.created_at)::text,'')
               FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
               WHERE ar.model_revision=%s AND ar.status IN ('complete','running')
                 AND EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.user_id=%s AND utl.track_id=mc.track_id)""",
            (MELODY_CONTOUR_REVISION, user_id),
        )
        row = cursor.fetchone()
    count, latest = row
    return f"{user_id}:{MELODY_CONTOUR_REVISION}", int(count), str(latest or "")


def _load_contours(connection: psycopg.Connection, user_id: uuid.UUID) -> list[dict[str, object]]:
    key = _contour_cache_key(connection, user_id)
    with _CACHE_LOCK:
        if _CONTOUR_CACHE["key"] == key and _CONTOUR_CACHE["contours"]:
            return _CONTOUR_CACHE["contours"]  # type: ignore[return-value]
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT DISTINCT ON (mc.track_id,mc.source)
                          mc.track_id,mc.source,mc.pitch,mc.voiced
                   FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
                   WHERE ar.status IN ('complete','running') AND ar.model_revision=%s
                     AND EXISTS (SELECT 1 FROM user_track_links utl
                                 WHERE utl.user_id=%s AND utl.track_id=mc.track_id)
                   ORDER BY mc.track_id,mc.source,ar.created_at DESC""",
                (MELODY_CONTOUR_REVISION, user_id),
            )
            rows = cursor.fetchall()
        contours = [
            {
                **row,
                "pitch": np.asarray(row["pitch"], dtype=np.float32),
                "voiced": np.asarray(row["voiced"], dtype=bool),
            }
            for row in rows
        ]
        _CONTOUR_CACHE["key"] = key
        _CONTOUR_CACHE["contours"] = contours
        return contours


def _capture_hum_diagnostic(
    user_id: uuid.UUID,
    audio: bytes,
    query: np.ndarray,
    query_mask: np.ndarray,
    results: list[dict[str, object]],
) -> str | None:
    root = Path(get_settings().hum_diagnostic_dir)
    marker = root.parent / "capture-next-hum"
    try:
        marker.unlink()
    except FileNotFoundError:
        return None
    diagnostic_id = str(uuid.uuid4())
    destination = root / diagnostic_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "recording.bin").write_bytes(audio)
    (destination / "diagnostic.json").write_text(
        json.dumps(
            {
                "id": diagnostic_id,
                "user_id": str(user_id),
                "matcher": "motif-rrf-batched-compiled-melody-dtw-v7",
                "query_pitch": query.tolist(),
                "query_voiced": query_mask.tolist(),
                "query_seconds": len(query) / CONTOUR_HZ,
                "voiced_ratio": float(query_mask.mean()),
                "results": [
                    {
                        "track_id": str(row["id"]),
                        "title": row["title"],
                        "artist": row.get("artist"),
                        "cost": row["match_cost"],
                        "similarity": row["similarity"],
                        "matched_at_seconds": row["matched_at_seconds"],
                        "matched_source": row["matched_source"],
                    }
                    for row in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return diagnostic_id


def _motif_windows(
    query: np.ndarray, query_mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the full query plus overlapping long and sparse short motifs."""
    plans = [(0, len(query))]
    if len(query) >= 90:
        plans.extend((start, 90) for start in range(0, len(query) - 89, 20))
    if len(query) >= 50:
        short_starts = list(range(0, len(query) - 49, 40))
        short_starts.append(((len(query) - 50) // 20) * 20)
        plans.extend((start, 50) for start in short_starts)
    plans = list(dict.fromkeys(plans))
    return [
        (query[start : start + width], query_mask[start : start + width]) for start, width in plans
    ]


def search_corpus(user_id: uuid.UUID, audio: bytes, limit: int = 10) -> dict[str, object]:
    query, query_mask = extract_hum_contour(audio)
    windows = _motif_windows(query, query_mask)
    prepared_motifs = _prepare_motifs(windows)
    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as connection:
        contours = _load_contours(connection, user_id)
        if not contours:
            raise ValueError("Build the melody hum index before searching")

        workers = min(get_settings().hum_max_workers, os.cpu_count() or 2)
        best_by_window: list[dict[uuid.UUID, tuple[float, float, str]]] = [{} for _ in windows]
        best_across_windows: dict[uuid.UUID, tuple[float, float, str]] = {}

        def scan(contour: dict[str, object]) -> tuple[uuid.UUID, str, list[tuple[float, float]]]:
            return (
                contour["track_id"],
                str(contour["source"]),
                _match_prepared_motifs(prepared_motifs, contour["pitch"], contour["voiced"]),
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for track_id, source, matches in pool.map(scan, contours):
                for motif_index, (cost, offset) in enumerate(matches):
                    if not np.isfinite(cost):
                        continue
                    current = best_by_window[motif_index].get(track_id)
                    if current is None or cost < current[0]:
                        best_by_window[motif_index][track_id] = (cost, offset, source)
                    overall = best_across_windows.get(track_id)
                    if overall is None or cost < overall[0]:
                        best_across_windows[track_id] = (cost, offset, source)

        rankings = [
            sorted(matches, key=lambda item: (matches[item][0], str(item)))
            for matches in best_by_window
        ]

        fusion: dict[uuid.UUID, float] = {}
        for ranking in rankings:
            for rank, track_id in enumerate(ranking, 1):
                fusion[track_id] = fusion.get(track_id, 0.0) + 1.0 / (20 + rank)
        ordered_ids = sorted(fusion, key=lambda item: (-fusion[item], str(item)))
        selected = [(track_id, *best_across_windows[track_id]) for track_id in ordered_ids[:limit]]
        if not selected:
            return {"tracks": [], "query_seconds": len(query) / CONTOUR_HZ}
        ids = [item[0] for item in selected]
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT t.id,t.title,t.artist,t.album,t.duration_seconds,max(utl.external_id) source_id,
                          max(ts.source_data->>'coverArt') cover_art
                   FROM tracks t JOIN user_track_links utl ON utl.track_id=t.id AND utl.user_id=%s
                   LEFT JOIN track_sources ts ON ts.library_id=utl.library_id AND ts.track_id=utl.track_id
                                              AND ts.external_id=utl.external_id
                   WHERE t.id=ANY(%s) GROUP BY t.id""",
                (user_id, ids),
            )
            metadata = {row["id"]: row for row in cursor.fetchall()}
    results = []
    for track_id, cost, offset, source in selected:
        row = dict(metadata[track_id])
        row.update(
            similarity=float(np.exp(-cost)),
            matched_at_seconds=offset,
            match_cost=cost,
            matched_source=source,
        )
        results.append(row)
    response = {
        "tracks": results,
        "query_seconds": len(query) / CONTOUR_HZ,
        "matcher": "motif-rrf-batched-compiled-melody-dtw-v7",
        "candidate_contours": len(contours),
        "query_windows": len(windows),
    }
    diagnostic_id = _capture_hum_diagnostic(user_id, audio, query, query_mask, results)
    if diagnostic_id:
        response["diagnostic_id"] = diagnostic_id
    return response
