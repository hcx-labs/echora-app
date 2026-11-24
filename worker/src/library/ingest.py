from __future__ import annotations

from core.db.config import DatabaseConfig
from core.embeddings.config import EmbeddingConfig

from dataclasses import dataclass
import hashlib
import json
import logging
import platform
import uuid
from collections.abc import Callable

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
import torch

from core.media.audio import decode_audio, full_coverage_window_ranges
from worker.library.descriptors import store_audio_descriptors
from worker.library.waveforms import store_waveform
from worker.library.visual_features import store_visual_features
from worker.melody.index import create_sync_run, store_track_contours
from core.embeddings.audio_models import (
    AudioEmbeddingModel,
    EmbeddingResult,
    MertModel,
    MuQMuLanModel,
    release_model,
)
from worker.remote.session import RemoteModel, current as current_remote
from core.integrations.navidrome import NavidromeClient, NavidromeTrack
from worker.planning.audio import plan_audio, audio_prerequisites
from worker.media.preprocessing import active_cache, prepare_audio, get_check
from core.embeddings.representations import configure_representations, embedding_config
from worker.library.attempts import start_attempt, record_track, finish_attempt
from worker.library.fingerprints import store_and_match_fingerprint

CONTENT_NAMESPACE = uuid.UUID("0c300f5d-99d1-48b8-b12a-a52b9556be86")
logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    requested: int = 0
    discovered: int = 0
    downloaded: int = 0
    inserted: int = 0
    already_linked: int = 0
    reused_embeddings: int = 0
    embedded_muq: int = 0
    embedded_mert: int = 0
    fingerprinted: int = 0
    waveforms_generated: int = 0
    visual_features_generated: int = 0
    melody_indexed: int = 0
    melody_contours: int = 0
    recording_matches: int = 0
    recording_fingerprinted: int = 0
    described_audio: int = 0
    failed: int = 0


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _config_hash(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _create_run(connection: psycopg.Connection, model: AudioEmbeddingModel) -> uuid.UUID:
    config = embedding_config(model.name, model.revision)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO analysis_runs
              (kind, model_name, model_revision, config_hash, config, environment,
               device, precision, status, started_at)
            VALUES ('audio_embedding', %s, %s, %s, %s, %s, %s, 'float32', 'running', now())
            ON CONFLICT (kind, model_name, model_revision, config_hash)
            DO UPDATE SET id = analysis_runs.id
            RETURNING id
            """,
            (
                model.name,
                model.revision,
                _config_hash(config),
                Jsonb(config),
                Jsonb(environment),
                str(model.device),
            ),
        )
        return cursor.fetchone()[0]


def _has_fingerprint(connection: psycopg.Connection, track_id: uuid.UUID) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM track_fingerprints WHERE track_id=%s", (track_id,))
        return cursor.fetchone() is not None


def _model_has_embedding(
    connection: psycopg.Connection, track_id: uuid.UUID, run_id: uuid.UUID
) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM embeddings WHERE track_id=%s AND run_id=%s AND embedding_type='audio-track'",
            (track_id, run_id),
        )
        return cursor.fetchone() is not None


def embed_track(
    audio: bytes,
    model: AudioEmbeddingModel,
) -> tuple[EmbeddingResult, list[tuple[float, float]]]:
    """Decode, window and embed one track. The same code runs locally and on Modal."""
    waveform = decode_audio(audio)
    ranged_windows = full_coverage_window_ranges(waveform)
    del waveform
    windows = [item[0] for item in ranged_windows]
    ranges = [(item[1], item[2]) for item in ranged_windows]
    return model.embed_windows(windows), ranges


def _store_embedding(
    connection: psycopg.Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    result: EmbeddingResult,
    ranges: list[tuple[float, float]] | None = None,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO embeddings
              (track_id, run_id, embedding_type, dimension, aggregation, embedding,
               inference_ms, peak_vram_bytes)
            VALUES (%s, %s, 'audio-track', %s, 'normalized_mean', %s::vector, %s, %s)
            ON CONFLICT (track_id, run_id, embedding_type, window_index) DO NOTHING
            """,
            (
                track_id,
                run_id,
                len(result.vector),
                _vector_literal(result.vector),
                result.inference_ms,
                result.peak_vram_bytes,
            ),
        )
        cursor.executemany(
            """
            INSERT INTO embeddings
              (track_id, run_id, embedding_type, window_index, window_start_seconds,
               window_end_seconds, dimension, aggregation, embedding)
            VALUES (%s, %s, 'audio-window', %s, %s, %s, %s, 'none', %s::vector)
            ON CONFLICT (track_id, run_id, embedding_type, window_index) DO NOTHING
            """,
            [
                (
                    track_id,
                    run_id,
                    window_index,
                    ranges[window_index][0] if ranges else None,
                    ranges[window_index][1] if ranges else None,
                    len(vector),
                    _vector_literal(vector),
                )
                for window_index, vector in enumerate(result.window_vectors)
            ],
        )


def _source_track_id(
    connection: psycopg.Connection, library_id: uuid.UUID, external_id: str
) -> uuid.UUID | None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT track_id FROM track_sources WHERE library_id=%s AND source_type='subsonic' AND external_id=%s",
            (library_id, external_id),
        )
        row = cursor.fetchone()
        return row[0] if row else None


def _upsert_track(
    connection: psycopg.Connection, library_id: uuid.UUID, song: NavidromeTrack, audio_hash: str
) -> tuple[uuid.UUID, bool]:
    from core.library.visibility import lock_library, source_remapped

    lock_library(connection, library_id)
    previous_id = _source_track_id(connection, library_id, song.id)
    track_id = uuid.uuid5(CONTENT_NAMESPACE, audio_hash)
    metadata = {"navidrome": song.raw, "identity": "sha256-source-bytes"}
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO tracks (id, audio_hash, title, artist, album, year, duration_seconds, genres, metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (audio_hash) DO NOTHING
            RETURNING id
            """,
            (
                track_id,
                audio_hash,
                song.title,
                song.artist,
                song.album,
                song.year,
                song.duration,
                [song.genre] if song.genre else [],
                Jsonb(metadata),
            ),
        )
        inserted = cursor.fetchone() is not None
        cursor.execute("SELECT id FROM tracks WHERE audio_hash=%s", (audio_hash,))
        canonical_id = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO library_tracks (library_id, track_id, relative_path) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
            (library_id, canonical_id, song.path),
        )
        cursor.execute(
            """
            INSERT INTO track_sources (library_id, track_id, source_type, external_id, source_data)
            VALUES (%s,%s,'subsonic',%s,%s)
            ON CONFLICT (library_id, source_type, external_id)
            DO UPDATE SET track_id=EXCLUDED.track_id, source_data=EXCLUDED.source_data
            """,
            (library_id, canonical_id, song.id, Jsonb(song.raw)),
        )
    if previous_id != canonical_id:
        source_remapped(connection, library_id, song.id)
    return canonical_id, inserted


def _library(connection: psycopg.Connection, url: str) -> uuid.UUID:
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/"))
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO libraries (name, root_path, namespace)
            VALUES ('Navidrome', %s, %s)
            ON CONFLICT (namespace) DO UPDATE SET root_path=EXCLUDED.root_path
            RETURNING id
            """,
            (url.rstrip("/"), namespace),
        )
        return cursor.fetchone()[0]


def _bound_source_audio(
    client: NavidromeClient, external_id: str, binding: tuple[uuid.UUID, str]
) -> tuple[bytes, uuid.UUID]:
    track_id, expected_hash = binding
    audio = client.audio_bytes(external_id)
    if hashlib.sha256(audio).hexdigest() != expected_hash:
        raise ValueError("Source content changed after identity resolution; retry this source")
    return audio, track_id


def ingest_navidrome(
    url: str,
    username: str,
    password: str,
    song_ids: list[str],
    progress: Callable[[dict[str, object]], None] | None = None,
    model_total: int = 2,
    *,
    verify_audio_hashes: bool = True,
) -> IngestSummary:
    report = progress or (lambda _: None)
    summary = IngestSummary(requested=len(song_ids))

    with (
        psycopg.connect(DatabaseConfig.load().database_url) as connection,
        NavidromeClient(url, username, password) as navidrome,
    ):
        configure_representations(connection)
        library_id = _library(connection, url)
        songs = navidrome.tracks(song_ids)
        summary.discovered = len(songs)
        bindings: dict[str, tuple[uuid.UUID, str]] = {}
        resolved_songs = []
        # Selected sources require an identity check independently of artifact
        # completeness. Bind later consumers to these exact bytes and IDs, even
        # when another worker remaps the source before this claim finishes.
        for song in songs:
            report({"phase": "identity", "message": "Resolving track identity"})
            try:
                get_check()()
                if not verify_audio_hashes:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """SELECT ts.track_id, t.audio_hash FROM track_sources ts
                            JOIN tracks t ON t.id=ts.track_id
                            WHERE ts.library_id=%s AND ts.source_type='subsonic' AND ts.external_id=%s""",
                            (library_id, song.id),
                        )
                        known = cursor.fetchone()
                        if known:
                            # Reuse identity, but do not claim the bytes were verified.
                            cursor.execute(
                                """UPDATE track_sources SET source_data=%s
                                WHERE library_id=%s AND source_type='subsonic' AND external_id=%s""",
                                (Jsonb(song.raw), library_id, song.id),
                            )
                            cursor.execute(
                                """UPDATE tracks SET title=%s, artist=%s, album=%s, year=%s,
                                duration_seconds=%s, genres=%s WHERE id=%s""",
                                (
                                    song.title,
                                    song.artist,
                                    song.album,
                                    song.year,
                                    song.duration,
                                    [song.genre] if song.genre else [],
                                    known[0],
                                ),
                            )
                    if known:
                        connection.commit()
                        bindings[song.id] = (known[0], known[1])
                        resolved_songs.append(song)
                        continue
                audio = navidrome.audio_bytes(song.id)
                audio_hash = hashlib.sha256(audio).hexdigest()
                track_id, inserted = _upsert_track(connection, library_id, song, audio_hash)
                with connection.cursor() as cursor:
                    cursor.execute(
                        """UPDATE track_sources SET audio_verified_at=now()
                        WHERE library_id=%s AND source_type='subsonic' AND external_id=%s
                            AND track_id=%s""",
                        (library_id, song.id, track_id),
                    )
                connection.commit()
                bindings[song.id] = (track_id, audio_hash)
                resolved_songs.append(song)
                summary.downloaded += 1
                summary.inserted += int(inserted)
                del audio
            except Exception:
                connection.rollback()
                summary.failed += 1
                logger.warning("Could not resolve source identity for Navidrome song %s", song.id)
        songs = resolved_songs
        plan = plan_audio(
            connection,
            library_id,
            [song.id for song in songs],
            resolved_track_ids={key: value[0] for key, value in bindings.items()},
        )
        required_models = int(plan.needs_muq) + int(plan.needs_mert)
        report(
            {
                "phase": "planning",
                "message": "Processing plan ready",
                "completed": 0,
                "total": len(plan.download_external_ids),
                "unit": "tracks",
                "plan": {
                    "muq": len(plan.muq_external_ids),
                    "mert": len(plan.mert_external_ids),
                    "fingerprint": len(plan.fingerprint_external_ids),
                    "recording_fingerprint": len(plan.recording_fingerprint_external_ids),
                    "melody": len(plan.melody_external_ids),
                    "descriptors": len(plan.descriptor_external_ids),
                    "waveform": len(plan.waveform_external_ids),
                    "visual_features": len(plan.visual_feature_external_ids),
                },
            }
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def audio_track(song: NavidromeTrack) -> tuple[bytes, uuid.UUID]:
            return _bound_source_audio(navidrome, song.id, bindings[song.id])

        # Prepare only the formats requested by outstanding tasks, before loading
        # GPU models. Same-format consumers and later batches read the disk cache.
        if active_cache() is not None:
            preparing = [(song, audio_prerequisites(plan, song.id)) for song in songs]
            preparing = [
                (song, needs) for song, needs in preparing if needs.mono_rates or needs.stereo_rates
            ]
            from worker.media.roformer import separation_phase

            # Separation phase for melody stems: Roformer loads once, and only if a song needs it.
            with separation_phase():
                for index, (song, needs) in enumerate(preparing):
                    report(
                        {
                            "phase": "preprocess",
                            "message": f"Preparing audio for {song.title}",
                            "completed": index,
                            "total": len(preparing),
                            "unit": "tracks",
                        }
                    )
                    try:
                        audio, _ = audio_track(song)
                        # Melody needs Roformer vocals; on Modal those are separated remotely.
                        prepare_audio(
                            audio,
                            mono_rates=needs.mono_rates,
                            stereo_rates=needs.stereo_rates,
                            melody=needs.melody and current_remote() is None,
                            check=get_check(),
                        )
                        del audio
                        connection.commit()
                    except Exception:
                        connection.rollback()
                        # Individual consumers retain their existing failure accounting
                        # and may retry; one bad source must not stop the whole batch.
                        logger.exception("Could not prepare audio for Navidrome song %s", song.id)
                    report(
                        {
                            "phase": "preprocess",
                            "message": f"Prepared audio for {song.title}",
                            "completed": index + 1,
                            "total": len(preparing),
                            "unit": "tracks",
                        }
                    )

        def embedding_phase(
            phase: str,
            label: str,
            external_ids: frozenset[str],
            model: AudioEmbeddingModel | RemoteModel,
        ) -> None:
            run_id = _create_run(connection, model)
            selected = [song for song in songs if song.id in external_ids]
            attempt_id = start_attempt(connection, run_id, len(selected))
            connection.commit()
            remote = current_remote()

            def started(index: int, total: int, song: NavidromeTrack) -> None:
                report(
                    {
                        "phase": phase,
                        "message": f"{label} {song.title}",
                        "track": {"id": song.id, "title": song.title, "artist": song.artist},
                        "completed": index,
                        "total": total,
                        "unit": "tracks",
                        "summary": summary.__dict__,
                    }
                )

            def finished(index: int, total: int, song: NavidromeTrack) -> None:
                report(
                    {
                        "phase": phase,
                        "message": f"{label} {song.title}",
                        "completed": index + 1,
                        "total": total,
                        "unit": "tracks",
                        "summary": summary.__dict__,
                    }
                )

            def reused(song: NavidromeTrack, track_id: uuid.UUID) -> bool:
                if not _model_has_embedding(connection, track_id, run_id):
                    return False
                summary.reused_embeddings += 1
                record_track(connection, attempt_id, song.id, track_id)
                connection.commit()
                return True

            def stored(
                song: NavidromeTrack,
                track_id: uuid.UUID,
                outcome: tuple[EmbeddingResult, list[tuple[float, float]]],
            ) -> None:
                _store_embedding(connection, track_id, run_id, *outcome)
                record_track(connection, attempt_id, song.id, track_id)
                if phase == "muq":
                    summary.embedded_muq += 1
                else:
                    summary.embedded_mert += 1
                connection.commit()

            def failed(song: NavidromeTrack) -> None:
                connection.rollback()
                summary.failed += 1
                logger.exception("Failed %s phase for Navidrome song %s", phase, song.id)
                # Do not persist provider exception messages, which may contain credentials.
                record_track(
                    connection,
                    attempt_id,
                    song.id,
                    error="Audio embedding failed; see service logs",
                )
                connection.commit()

            if remote is None:
                for index, song in enumerate(selected):
                    started(index, len(selected), song)
                    try:
                        audio, track_id = audio_track(song)
                        if reused(song, track_id):
                            continue
                        outcome = embed_track(audio, model)
                        del audio
                        stored(song, track_id, outcome)
                    except Exception:
                        failed(song)
                    finished(index, len(selected), song)
            else:
                # Upload each track's audio once, then embed the whole phase in one
                # Modal call. Results stream back in order and are stored as they arrive.
                pending: list[tuple[NavidromeTrack, uuid.UUID, str]] = []
                for song in selected:
                    try:
                        audio, track_id = audio_track(song)
                        if reused(song, track_id):
                            continue
                        remote.upload(bindings[song.id][1], audio)
                        del audio
                        pending.append((song, track_id, bindings[song.id][1]))
                    except Exception:
                        failed(song)
                outcomes = remote.embed_audio(model, [digest for _, _, digest in pending])
                for index, ((song, track_id, _), outcome) in enumerate(zip(pending, outcomes)):
                    started(index, len(pending), song)
                    try:
                        if isinstance(outcome, BaseException):
                            raise outcome
                        stored(song, track_id, outcome)
                    except Exception:
                        failed(song)
                    finished(index, len(pending), song)
            finish_attempt(connection, attempt_id)
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE analysis_runs SET status='complete', finished_at=now() WHERE id=%s",
                    (run_id,),
                )
            connection.commit()

        waveform_songs = [song for song in songs if song.id in plan.waveform_external_ids]
        for index, song in enumerate(waveform_songs):
            report(
                {
                    "phase": "waveform",
                    "message": f"Generating waveform for {song.title}",
                    "completed": index,
                    "total": len(waveform_songs),
                    "unit": "tracks",
                }
            )
            try:
                audio, track_id = audio_track(song)
                store_waveform(connection, track_id, audio)
                del audio
                connection.commit()
                summary.waveforms_generated += 1
            except Exception:
                connection.rollback()
                summary.failed += 1
                logger.exception("Could not generate waveform for Navidrome song %s", song.id)
            report(
                {
                    "phase": "waveform",
                    "message": f"Generating waveform for {song.title}",
                    "completed": index + 1,
                    "total": len(waveform_songs),
                    "unit": "tracks",
                    "summary": summary.__dict__,
                }
            )

        visual_feature_songs = [
            song for song in songs if song.id in plan.visual_feature_external_ids
        ]
        for index, song in enumerate(visual_feature_songs):
            report(
                {
                    "phase": "visual_features",
                    "message": f"Preparing visual features for {song.title}",
                    "completed": index,
                    "total": len(visual_feature_songs),
                    "unit": "tracks",
                }
            )
            try:
                audio, track_id = audio_track(song)
                store_visual_features(connection, track_id, audio)
                del audio
                connection.commit()
                summary.visual_features_generated += 1
            except Exception:
                connection.rollback()
                summary.failed += 1
                logger.exception("Could not prepare visual features for Navidrome song %s", song.id)
            report(
                {
                    "phase": "visual_features",
                    "message": f"Prepared visual features for {song.title}",
                    "completed": index + 1,
                    "total": len(visual_feature_songs),
                    "unit": "tracks",
                    "summary": summary.__dict__,
                }
            )

        loaded = 0
        if plan.needs_muq:
            report(
                {
                    "phase": "models",
                    "message": "Loading semantic model",
                    "completed": loaded,
                    "total": required_models,
                    "unit": "models",
                }
            )
            muq = (
                RemoteModel(
                    MuQMuLanModel.name,
                    EmbeddingConfig.load().muq_model_id,
                    EmbeddingConfig.load().muq_revision,
                )
                if current_remote()
                else MuQMuLanModel(
                    EmbeddingConfig.load().muq_model_id, EmbeddingConfig.load().muq_revision, device
                )
            )
            try:
                embedding_phase("muq", "Embedding semantics for", plan.muq_external_ids, muq)
            finally:
                release_model(muq)
                del muq
            loaded += 1

        if plan.needs_mert:
            report(
                {
                    "phase": "models",
                    "message": "Loading acoustic model",
                    "completed": loaded,
                    "total": required_models,
                    "unit": "models",
                }
            )
            mert = (
                RemoteModel(
                    MertModel.name,
                    EmbeddingConfig.load().mert_model_id,
                    EmbeddingConfig.load().mert_revision,
                )
                if current_remote()
                else MertModel(
                    EmbeddingConfig.load().mert_model_id,
                    EmbeddingConfig.load().mert_revision,
                    device,
                )
            )
            try:
                embedding_phase("mert", "Embedding acoustics for", plan.mert_external_ids, mert)
            finally:
                release_model(mert)
                del mert

        melody_songs = [song for song in songs if song.id in plan.melody_external_ids]
        if melody_songs:
            melody_run_id = create_sync_run(connection)
            connection.commit()
            remote = current_remote()
            unavailable: dict[str, BaseException] = {}
            if remote is not None:
                # Upload once, extract every track's contours in one Modal call;
                # results stream back in song order and are stored as they arrive.
                pending = []
                for song in melody_songs:
                    try:
                        audio, _ = audio_track(song)
                        remote.upload(bindings[song.id][1], audio)
                        pending.append(song)
                    except Exception as error:
                        unavailable[song.id] = error
                # Separation phase on Modal first: one Roformer for every melody song.
                separated = remote.prepare_vocals(
                    [bindings[song.id][1] for song in pending], melody=True
                )
                for song, outcome in zip(list(pending), separated):
                    if isinstance(outcome, BaseException):
                        unavailable[song.id] = outcome
                        pending.remove(song)
                outcomes = iter(remote.melody([bindings[song.id][1] for song in pending]))
            for index, song in enumerate(melody_songs):
                report(
                    {
                        "phase": "melody",
                        "message": f"Extracting melody from {song.title}",
                        "completed": index,
                        "total": len(melody_songs),
                        "unit": "tracks",
                        "summary": summary.__dict__,
                    }
                )
                try:
                    if remote is None:
                        audio, track_id = audio_track(song)
                        stored = store_track_contours(connection, track_id, melody_run_id, audio)
                    else:
                        outcome = unavailable.get(song.id) or next(outcomes)
                        if isinstance(outcome, BaseException):
                            raise outcome
                        stored = store_track_contours(
                            connection, bindings[song.id][0], melody_run_id, b"", usable=outcome
                        )
                    summary.melody_contours += stored
                    summary.melody_indexed += 1
                    connection.commit()
                except Exception:
                    connection.rollback()
                    summary.failed += 1
                    logger.exception("Could not extract melody for Navidrome song %s", song.id)
                report(
                    {
                        "phase": "melody",
                        "message": f"Extracting melody from {song.title}",
                        "completed": index + 1,
                        "total": len(melody_songs),
                        "unit": "tracks",
                        "summary": summary.__dict__,
                    }
                )
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE analysis_runs SET status='complete',finished_at=now() WHERE id=%s",
                    (melody_run_id,),
                )
            connection.commit()

        fingerprint_songs = [song for song in songs if song.id in plan.fingerprint_external_ids]
        for index, song in enumerate(fingerprint_songs):
            report(
                {
                    "phase": "fingerprint",
                    "message": f"Fingerprinting {song.title}",
                    "track": {"id": song.id, "title": song.title, "artist": song.artist},
                    "completed": index,
                    "total": len(fingerprint_songs),
                    "unit": "tracks",
                    "summary": summary.__dict__,
                }
            )
            try:
                audio, track_id = audio_track(song)
                with connection.transaction():
                    match = store_and_match_fingerprint(connection, track_id, audio, song.duration)
                summary.fingerprinted += 1
                summary.recording_matches += int(bool(match.get("matched")))
                del audio
                connection.commit()
            except Exception:
                connection.rollback()
                summary.failed += 1
                logger.exception("Could not fingerprint Navidrome song %s", song.id)
            report(
                {
                    "phase": "fingerprint",
                    "message": f"Fingerprinting {song.title}",
                    "completed": index + 1,
                    "total": len(fingerprint_songs),
                    "unit": "tracks",
                    "summary": summary.__dict__,
                }
            )

        recording_songs = [
            song for song in songs if song.id in plan.recording_fingerprint_external_ids
        ]
        if recording_songs:
            from worker.recordings.encoder import config_from_env
            from worker.recordings.search import store_fingerprints

            recording_config = config_from_env()
            if recording_config is None:
                raise ValueError("Recording encoder configuration disappeared during analysis")
            for index, song in enumerate(recording_songs):
                report(
                    {
                        "phase": "recording_fingerprint",
                        "message": f"Indexing recording for {song.title}",
                        "completed": index,
                        "total": len(recording_songs),
                        "unit": "tracks",
                    }
                )
                failed = False
                try:
                    audio, track_id = audio_track(song)
                    stored = store_fingerprints(
                        connection, track_id, audio, recording_config, get_check()
                    )
                    del audio
                    connection.commit()
                    summary.recording_fingerprinted += int(stored)
                except Exception:
                    connection.rollback()
                    summary.failed += 1
                    failed = True
                    logger.error("Could not index recording for Navidrome song %s", song.id)
                report(
                    {
                        "phase": "recording_fingerprint",
                        "message": (
                            f"Recording indexing failed for {song.title}"
                            if failed
                            else f"Recording indexing finished for {song.title}"
                        ),
                        "completed": index + 1,
                        "total": len(recording_songs),
                        "unit": "tracks",
                        "summary": summary.__dict__,
                    }
                )

        descriptor_songs = [song for song in songs if song.id in plan.descriptor_external_ids]
        for index, song in enumerate(descriptor_songs):
            try:
                audio, track_id = audio_track(song)
                descriptors = store_audio_descriptors(connection, track_id, audio)
                del audio
                if descriptors.get("failed_components"):
                    summary.failed += 1
                else:
                    summary.described_audio += 1
                connection.commit()
            except Exception:
                connection.rollback()
                summary.failed += 1
                logger.exception(
                    "Could not analyze sound descriptors for Navidrome song %s", song.id
                )
            report(
                {
                    "phase": "audio_descriptors",
                    "message": f"Measuring sound for {song.title}",
                    "completed": index + 1,
                    "total": len(descriptor_songs),
                    "unit": "tracks",
                    "summary": summary.__dict__,
                }
            )

        summary.already_linked = len(songs) - len(plan.download_external_ids)
        report(
            {
                "phase": "finalizing",
                "message": "Analysis phases complete",
                "completed": len(songs),
                "total": len(songs),
                "unit": "tracks",
            }
        )
    return summary
