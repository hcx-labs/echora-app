from __future__ import annotations

from .settings import get_settings

from collections.abc import Callable
import hashlib
import json
import logging
import platform
import uuid

import psycopg
from psycopg.types.json import Jsonb
import torch

from .lyrics_analysis import LyricsEmbeddingModel
from .language_detection import detect_distribution
from .models import release_model
from .navidrome import NavidromeClient
from .navidrome_lyrics_sources import store_original
from .processing_plan import plan_lyrics, resolve_library_id
from .representations import configure_representations, embedding_config
from .analysis_attempts import start_attempt, record_track, finish_attempt
from .preprocessing import prepare_audio, get_check
from .remote_compute import RemoteModel, current as current_remote
from .roformer import separation_phase
from .song_transcription import SongTranscriber, needs_repair
from .transcription_recovery import diagnostic_writer


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _create_run(connection: psycopg.Connection, model: LyricsEmbeddingModel) -> uuid.UUID:
    config = embedding_config(model.name, model.revision)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO analysis_runs
                 (kind, model_name, model_revision, config_hash, config, environment, device, precision, status, started_at)
               VALUES ('lyrics_embedding',%s,%s,%s,%s,%s,%s,'float32','running',now())
               ON CONFLICT (kind, model_name, model_revision, config_hash)
               DO UPDATE SET id=analysis_runs.id RETURNING id""",
            (
                model.name,
                model.revision,
                config_hash,
                Jsonb(config),
                Jsonb(environment),
                str(model.device),
            ),
        )
        return cursor.fetchone()[0]


def _provenance(result: dict[str, object], language_distribution: dict) -> dict[str, object]:
    return {
        "provider": "moss" if result.get("ai_generated") else "navidrome",
        "endpoint": result.get("source"),
        "ai_generated": bool(result.get("ai_generated")),
        **({"transcription": result["transcription"]} if result.get("transcription") else {}),
        "synced": result.get("synced"),
        "lines": result.get("lines") or [],
        **({"languages": language_distribution} if language_distribution else {}),
    }


def _language(language_distribution: dict, result: dict[str, object]) -> str | None:
    language = language_distribution.get("primary_language") or result.get("language")
    return None if language == "xxx" else language


def _refine_transcripts(connection, items, summary, report) -> None:
    """External AI second pass over this run's new AI transcripts, in place in `items`.

    Runs after transcription has released its models and before embedding, so search,
    translations and karaoke all use the corrected text. A failure keeps the transcript.
    """
    from . import main
    from .transcription_refinement import RefinementError, enabled, refine
    from .translation_storage import load_settings

    try:
        settings, encrypted = load_settings()
    except Exception:
        logging.getLogger(__name__).exception(
            "Could not load External AI settings; transcripts stay unrefined"
        )
        return
    if not items or not enabled(settings):
        return
    key = main._cipher().decrypt(bytes(encrypted)).decode() if encrypted else None
    for index, (position, track_id, title, lyrics) in enumerate(items):
        report(
            {
                "phase": "lyrics",
                "message": f"Refining transcript for {title}",
                "completed": index,
                "total": len(items),
                "unit": "tracks",
            }
        )
        artist = connection.execute("SELECT artist FROM tracks WHERE id=%s", (track_id,)).fetchone()
        try:
            refined = refine(
                settings,
                key,
                lyrics,
                title=title,
                artist=(artist[0] if artist else "") or "",
                languages=(detect_distribution(str(lyrics["text"])).get("languages") or {}),
            )
        except RefinementError as error:
            summary["refinement_failed"] = summary.get("refinement_failed", 0) + 1
            logging.getLogger(__name__).warning(
                "Transcript refinement failed for %s: %s", track_id, error
            )
            continue
        distribution = detect_distribution(str(refined["text"]))
        updated = connection.execute(
            """UPDATE lyrics SET text=%s, language=%s, provenance=%s
               WHERE track_id=%s AND source='transcribed' AND text=%s""",
            (
                refined["text"],
                _language(distribution, refined),
                Jsonb(_provenance(refined, distribution)),
                track_id,
                lyrics["text"],
            ),
        ).rowcount
        connection.commit()
        if updated:
            summary["refined"] = summary.get("refined", 0) + 1
            items[index] = (position, track_id, title, refined)


def _store_lyrics(
    connection: psycopg.Connection, track_id: uuid.UUID, result: dict[str, object]
) -> uuid.UUID | None:
    text = result.get("text")
    status = str(result.get("status") or "unavailable")
    source = "transcribed" if result.get("ai_generated") else "embedded" if text else "none"
    language_distribution = detect_distribution(str(text) if text else None)
    language = _language(language_distribution, result)
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO lyrics (track_id, source, text, language, provenance, availability_status)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (track_id) DO UPDATE SET source=EXCLUDED.source, text=EXCLUDED.text,
                 language=EXCLUDED.language, provenance=EXCLUDED.provenance,
                 availability_status=EXCLUDED.availability_status, created_at=now()
               WHERE (EXCLUDED.source != 'transcribed' AND EXCLUDED.text IS NOT NULL)
                  OR NULLIF(btrim(lyrics.text), '') IS NULL
               RETURNING id""",
            (
                track_id,
                source,
                text,
                language,
                Jsonb(_provenance(result, language_distribution)),
                status,
            ),
        )
        row = cursor.fetchone()
        return row[0] if row else None


def _store_embeddings(
    connection: psycopg.Connection, track_id: uuid.UUID, run_id: uuid.UUID, result
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM embeddings WHERE track_id=%s AND run_id=%s AND embedding_type='lyrics'",
            (track_id, run_id),
        )
        cursor.execute(
            """INSERT INTO embeddings
                 (track_id, run_id, embedding_type, dimension, aggregation, embedding, inference_ms, peak_vram_bytes)
               VALUES (%s,%s,'lyrics',%s,'normalized_mean',%s::vector,%s,%s)""",
            (
                track_id,
                run_id,
                len(result.aggregate),
                _vector_literal(result.aggregate),
                result.inference_ms,
                result.peak_vram_bytes,
            ),
        )
        for index, (vector, token_range) in enumerate(zip(result.windows, result.token_ranges)):
            cursor.execute(
                """INSERT INTO embeddings
                     (track_id, run_id, embedding_type, window_index, dimension, aggregation, embedding)
                   VALUES (%s,%s,'lyrics',%s,%s,%s,%s::vector)""",
                (
                    track_id,
                    run_id,
                    index,
                    len(vector),
                    f"tokens:{token_range[0]}-{token_range[1]}",
                    _vector_literal(vector),
                ),
            )


def backfill_lyrics(
    url: str,
    username: str,
    password: str,
    progress: Callable[[dict[str, object]], None] | None = None,
    external_ids: list[str] | None = None,
    only_missing: bool = False,
    refresh_existing: bool = False,
) -> dict[str, int]:
    report = progress or (lambda _: None)
    summary = {
        "total": 0,
        "available": 0,
        "missing": 0,
        "unavailable": 0,
        "embedded": 0,
        "failed": 0,
    }
    with (
        psycopg.connect(get_settings().database_url) as connection,
        NavidromeClient(url, username, password) as client,
    ):
        library_id = resolve_library_id(connection, url)
        configure_representations(connection)
        planned = plan_lyrics(connection, external_ids, library_id=library_id).lyrics_external_ids
        if refresh_existing and external_ids is not None:
            planned = frozenset(external_ids)
        if not planned:
            report(
                {
                    "phase": "planning",
                    "message": "Lyrics analysis already current",
                    "completed": 0,
                    "total": 0,
                    "unit": "tracks",
                }
            )
            return summary
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT DISTINCT ON (ts.track_id) ts.track_id, ts.external_id, t.title
                   FROM track_sources ts JOIN tracks t ON t.id=ts.track_id
                   WHERE ts.source_type='subsonic' AND ts.external_id=ANY(%s) AND ts.library_id=%s
                   ORDER BY ts.track_id, ts.id""",
                (list(planned), library_id),
            )
            tracks = cursor.fetchall()
        summary["total"] = len(tracks)
        embeddable: list[tuple[uuid.UUID, str, dict[str, object]]] = []
        fresh_transcripts: set[uuid.UUID] = set()
        transcriptions = []

        def accept_lyrics(track_id, title, lyrics, stored=False):
            if not stored:
                stored_id = _store_lyrics(connection, track_id, lyrics)
                if stored_id is None:
                    # A concurrent writer supplied text. Never embed our discarded candidate.
                    connection.commit()
                    return
            status = str(lyrics.get("status") or "unavailable")
            summary[status] = summary.get(status, 0) + 1
            if lyrics.get("text"):
                embeddable.append((track_id, title, lyrics))
                if not stored and lyrics.get("ai_generated") and lyrics.get("lines"):
                    fresh_transcripts.add(track_id)
            connection.commit()

        for index, (track_id, external_id, title) in enumerate(tracks):
            try:
                provider_lyrics = client.lyrics(external_id)
                store_original(connection, library_id, external_id, track_id, provider_lyrics)
                connection.commit()
                stored = None
                # Archive the provider separately; manual edits remain authoritative
                # when their embedding needs to be rebuilt.
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT text, provenance, availability_status, source FROM lyrics WHERE track_id=%s",
                        (track_id,),
                    )
                    stored = cursor.fetchone()
                if stored and (stored[1] or {}).get("manual_status"):
                    # A user-set status is final, even though these rows use source='none'.
                    lyrics = {
                        "text": stored[0],
                        "status": (stored[1] or {})["manual_status"],
                        **(stored[1] or {}),
                    }
                elif stored and stored[3] == "manual":
                    lyrics = {"text": stored[0], "status": stored[2], **(stored[1] or {})}
                elif stored and (stored[1] or {}).get("forced_transcription"):
                    forced_language = str((stored[1] or {}).get("transcription_language") or "")
                    from .transcription_config import transcription_model, transcription_enabled

                    config = transcription_model() if transcription_enabled(connection) else None
                    if not config:
                        raise RuntimeError("AI lyric generation is disabled")
                    transcriptions.append(
                        (
                            track_id,
                            external_id,
                            title,
                            config,
                            {"status": "missing", "transcription_language": forced_language},
                        )
                    )
                    stored = None
                    connection.commit()
                    report(
                        {
                            "phase": "lyrics",
                            "message": f"Lyrics need transcription for {title}",
                            "completed": index + 1,
                            "total": len(tracks),
                            "unit": "tracks",
                        }
                    )
                    continue
                else:
                    stored = None
                    lyrics = provider_lyrics
                    # A retrieval miss must not erase a previous AI transcript or lyrics
                    # that merely need embedding with a newer embedding model.
                    if not str(lyrics.get("text") or "").strip():
                        with connection.cursor() as cursor:
                            cursor.execute(
                                "SELECT text, provenance, availability_status FROM lyrics WHERE track_id=%s AND (NULLIF(btrim(text),'') IS NOT NULL OR availability_status='instrumental')",
                                (track_id,),
                            )
                            stored = cursor.fetchone()
                        if stored:
                            lyrics = {"text": stored[0], "status": stored[2], **(stored[1] or {})}
                        elif lyrics.get("status") != "instrumental":
                            from .transcription_config import (
                                transcription_model,
                                transcription_enabled,
                            )

                            config = (
                                transcription_model() if transcription_enabled(connection) else None
                            )
                            if config:
                                transcriptions.append(
                                    (track_id, external_id, title, config, lyrics)
                                )
                                connection.commit()
                                report(
                                    {
                                        "phase": "lyrics",
                                        "message": f"Lyrics need transcription for {title}",
                                        "completed": index + 1,
                                        "total": len(tracks),
                                        "unit": "tracks",
                                    }
                                )
                                continue
                accept_lyrics(track_id, title, lyrics, stored=bool(stored))
            except Exception:
                connection.rollback()
                logging.getLogger(__name__).exception(
                    "Lyrics retrieval/transcription failed for %s", track_id
                )
                summary["failed"] += 1
            report(
                {
                    "phase": "lyrics",
                    "message": f"Retrieving lyrics for {title}",
                    "completed": index + 1,
                    "total": len(tracks),
                    "unit": "tracks",
                    "summary": summary,
                }
            )

        # Only genuine fallbacks reach this stage. Never separate songs solely
        # because their supplied lyrics need a new embedding.
        # Phases (docs/batch-phases.md): separate every selected song's vocals with one
        # Roformer, generate every song with MOSS loaded once, then repair compressed
        # timing with the karaoke aligner loaded once, only for drafts that need it.
        remote = current_remote()
        ready = []
        digests = {}
        with separation_phase():
            for index, candidate in enumerate(transcriptions):
                track_id, external_id, title, config, missing = candidate
                report(
                    {
                        "phase": "preprocess",
                        "message": f"Preparing vocals for {title}",
                        "completed": index,
                        "total": len(transcriptions),
                        "unit": "tracks",
                    }
                )
                try:
                    from .transcription_config import transcription_enabled

                    if not transcription_enabled(connection):
                        accept_lyrics(track_id, title, missing)
                        continue
                    if remote is None:
                        prepare_audio(
                            client.audio_bytes(external_id), vocals=True, check=get_check()
                        )
                    else:
                        # Vocals are separated on Modal; only the source audio goes up.
                        digests[track_id] = remote.upload_audio(client.audio_bytes(external_id))
                    ready.append(candidate)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    summary["failed"] += 1
                    logging.getLogger(__name__).exception(
                        "Vocal preparation failed for %s", track_id
                    )
                report(
                    {
                        "phase": "preprocess",
                        "completed": index + 1,
                        "total": len(transcriptions),
                        "unit": "tracks",
                        "summary": summary,
                    }
                )
        if remote is not None and ready:
            separated = remote.prepare_vocals([digests[item[0]] for item in ready], vocals=True)
            kept = []
            for candidate, outcome in zip(list(ready), separated):
                if isinstance(outcome, BaseException):
                    summary["failed"] += 1
                    logging.getLogger(__name__).error(
                        "Vocal separation on Modal failed for %s", candidate[0]
                    )
                else:
                    kept.append(candidate)
            ready = kept

        transcribers: dict[tuple, SongTranscriber] = {}
        repairs = []
        try:
            for index, (track_id, external_id, title, config, missing) in enumerate(ready):
                try:
                    from .transcription_config import transcription_enabled

                    if not transcription_enabled(connection):
                        accept_lyrics(track_id, title, missing)
                        continue
                    transcriber = transcribers.get(tuple(config))
                    if transcriber is None:
                        transcriber = transcribers[tuple(config)] = SongTranscriber(*config)
                        if remote is None:
                            # MOSS loads once for every song of this phase.
                            transcriber.open()
                    report(
                        {
                            "phase": "transcription",
                            "message": f"Transcribing lyrics for {title}",
                            "completed": index,
                            "total": len(ready),
                            "unit": "tracks",
                        }
                    )
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """SELECT a.activity FROM track_vocal_activity a
                            JOIN current_embeddings e ON e.track_id=a.track_id AND e.run_id=a.run_id
                            WHERE a.track_id=%s AND e.embedding_type='voice-gender'
                            ORDER BY a.created_at DESC LIMIT 1""",
                            (track_id,),
                        )
                        activity_row = cursor.fetchone()
                    arguments = dict(
                        language=str(missing.get("transcription_language") or "") or None,
                        check=lambda: report({"phase": "transcription"}),
                        progress=lambda detail, title=title: report(
                            {"phase": "transcription", "message": f"{title}: {detail}"}
                        ),
                        diagnostic_sink=diagnostic_writer(track_id),
                        vocal_activity=activity_row[0] if activity_row else None,
                    )
                    if remote is None:
                        draft = transcriber.generate(client.audio_bytes(external_id), **arguments)
                    else:
                        draft = remote.transcribe_generate(
                            client.audio_bytes(external_id), **arguments
                        )
                    if needs_repair(draft):
                        # Saved after the timing repair phase.
                        repairs.append((track_id, title, transcriber, draft))
                    else:
                        accept_lyrics(track_id, title, transcriber.finish(draft))
                except Exception:
                    connection.rollback()
                    summary["failed"] += 1
                    logging.getLogger(__name__).exception(
                        "Lyrics transcription failed for %s", track_id
                    )
                report(
                    {
                        "phase": "transcription",
                        "message": f"Processed lyrics for {title}",
                        "completed": index + 1,
                        "total": len(ready),
                        "unit": "tracks",
                        "summary": summary,
                    }
                )
        finally:
            for transcriber in transcribers.values():
                transcriber.close()

        if repairs:
            # Timing repair phase: the karaoke aligner, loaded once, for drafts that need it.
            finished = (
                iter(remote.transcribe_finish([item[3] for item in repairs]))
                if remote is not None
                else None
            )
            try:
                for index, (track_id, title, transcriber, draft) in enumerate(repairs):
                    report(
                        {
                            "phase": "transcription",
                            "message": f"Repairing timing for {title}",
                            "completed": index,
                            "total": len(repairs),
                            "unit": "tracks",
                        }
                    )
                    try:
                        lyrics = transcriber.finish(draft) if finished is None else next(finished)
                        if isinstance(lyrics, BaseException):
                            raise lyrics
                        accept_lyrics(track_id, title, lyrics)
                    except Exception:
                        connection.rollback()
                        summary["failed"] += 1
                        logging.getLogger(__name__).exception(
                            "Lyrics timing repair failed for %s", track_id
                        )
            finally:
                if finished is None:
                    from .karaoke_pipeline import _stop_fa_kara_worker

                    _stop_fa_kara_worker()

        fresh = [(i, *item) for i, item in enumerate(embeddable) if item[0] in fresh_transcripts]
        if fresh:
            _refine_transcripts(connection, fresh, summary, report)
            for position, track_id, title, lyrics in fresh:
                embeddable[position] = (track_id, title, lyrics)
        if not embeddable:
            return summary
        device = "cuda" if torch.cuda.is_available() else "cpu"
        report(
            {
                "phase": "models",
                "message": "Loading BGE-M3 lyrics model",
                "completed": 0,
                "total": 1,
                "unit": "models",
            }
        )
        remote = current_remote()
        model = (
            RemoteModel(
                LyricsEmbeddingModel.name,
                get_settings().lyrics_model_id,
                get_settings().lyrics_revision,
            )
            if remote
            else LyricsEmbeddingModel(
                get_settings().lyrics_model_id, get_settings().lyrics_revision, device
            )
        )
        try:
            run_id = _create_run(connection, model)
            attempt_id = start_attempt(connection, run_id, len(embeddable))
            connection.commit()
            # On Modal, every text is embedded in one call; results stream back in order.
            remote_outcomes = (
                iter(remote.embed_lyrics([str(item[2]["text"]) for item in embeddable]))
                if remote
                else None
            )
            for index, (track_id, title, lyrics) in enumerate(embeddable):
                try:
                    if remote_outcomes is None:
                        embedded = model.embed(str(lyrics["text"]))
                    else:
                        embedded = next(remote_outcomes)
                        if isinstance(embedded, BaseException):
                            raise embedded
                    _store_embeddings(connection, track_id, run_id, embedded)
                    record_track(connection, attempt_id, str(track_id), track_id)
                    summary["embedded"] += 1
                    connection.commit()
                except Exception:
                    connection.rollback()
                    summary["failed"] += 1
                    record_track(
                        connection,
                        attempt_id,
                        str(track_id),
                        track_id,
                        error="Lyrics embedding failed; see service logs",
                    )
                    connection.commit()
                report(
                    {
                        "phase": "lyrics",
                        "message": f"Embedding lyrics for {title}",
                        "completed": index + 1,
                        "total": len(embeddable),
                        "unit": "tracks",
                        "summary": summary,
                    }
                )
            finish_attempt(connection, attempt_id)
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE analysis_runs SET status='complete', finished_at=now() WHERE id=%s",
                    (run_id,),
                )
            connection.commit()
        finally:
            release_model(model)
            del model
    return summary
