"""Subprocess analysis orchestration. Queue payloads contain references, never secrets.

Parents: connection_id + optional external track_ids; import requires track_ids.
Profiles instead use canonical UUID track_ids and require no connection_id.
Hum parents accept track_limit and create a durable corpus using the parent job ID.
Hum children carry corpus_id; parent coordination finalizes corpus status.
Children: the same payload plus operation (parent kind), bounded track_ids.
User identity belongs to the job envelope, not the payload.
"""

from __future__ import annotations

from .settings import get_settings, require_database_url

from contextlib import ExitStack
from dataclasses import asdict
import logging
import uuid

import psycopg

from .navidrome import NavidromeClient, batch_audio_cache
from .preprocessing import preprocessing_session
from . import remote_compute

logger = logging.getLogger(__name__)

OPERATIONS = frozenset(
    {
        "navidrome_sync",
        "import",
        "lyrics_backfill",
        "voice_backfill",
        "karaoke_backfill",
        "recordings_backfill",
        "audio_profiles",
        "hum_corpus",
        "semantic_fusion_build",
    }
)


def _connect():
    return psycopg.connect(require_database_url())


def _source_rows(user_id, url, ids=None):
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT DISTINCT ts.external_id, ts.track_id FROM track_sources ts
               JOIN libraries l ON l.id=ts.library_id
               JOIN user_track_links u ON u.library_id=ts.library_id AND u.track_id=ts.track_id
                 AND u.external_id=ts.external_id
               WHERE u.user_id=%s AND l.root_path=%s AND ts.source_type='subsonic'
                 AND (%s::text[] IS NULL OR ts.external_id=ANY(%s)) ORDER BY ts.external_id""",
            (user_id, url.rstrip("/"), ids, ids),
        )
        return cursor.fetchall()


def _add_links(user_id, url, ids):
    from .source_visibility import update_memberships

    # Never reconcile a batch: removing absent IDs would delete sibling batches.
    with _connect() as connection, connection.cursor() as cursor:
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/"))
        cursor.execute("SELECT id FROM libraries WHERE namespace=%s", (namespace,))
        library = cursor.fetchone()
        if library is None:
            return 0
        return update_memberships(connection, library[0], user_id, ids)["linked"]


def _profiles(ids, report):
    from .audio_profiles import SUPPORTED_PROFILE_MODELS, build_audio_profiles

    models = {
        name: build_audio_profiles([uuid.UUID(str(v)) for v in ids], report, model_name=name)
        for name in SUPPORTED_PROFILE_MODELS
    }
    return {
        "models": models,
        "profiled": sum(v["profiled"] for v in models.values()),
        "failed": sum(v["failed"] for v in models.values()),
    }


def _recordings(credentials, user_id, ids, report, check):
    from .recordings import store_and_match_fingerprint

    summary = {"fingerprinted": 0, "matched": 0, "failed": 0}
    with (
        _connect() as connection,
        connection.cursor() as cursor,
        NavidromeClient(*credentials) as client,
    ):
        for external_id, track_id in _source_rows(user_id, credentials[0], ids):
            check()
            cursor.execute(
                "SELECT duration_seconds FROM tracks WHERE id=%s AND NOT EXISTS (SELECT 1 FROM track_fingerprints WHERE track_id=%s)",
                (track_id, track_id),
            )
            row = cursor.fetchone()
            if not row:
                continue
            try:
                result = store_and_match_fingerprint(
                    connection, track_id, client.audio_bytes(external_id), float(row[0] or 0)
                )
                connection.commit()
                summary["fingerprinted"] += 1
                summary["matched"] += int(bool(result.get("matched")))
            except Exception:
                connection.rollback()
                summary["failed"] += 1
            report({"phase": "fingerprint", "summary": dict(summary)})
    return summary


def _hum(credentials, user_id, payload, report, check):
    from .hum_search import create_sync_run, store_track_contours
    from .preprocessing import prepare_audio

    summary = {"tracks": 0, "contours": 0, "failed": 0}
    with (
        _connect() as connection,
        connection.cursor() as cursor,
        NavidromeClient(*credentials) as client,
    ):
        cursor.execute(
            "SELECT id FROM hum_corpora WHERE id=%s AND user_id=%s", (payload["corpus_id"], user_id)
        )
        if not cursor.fetchone():
            raise ValueError("Unknown hum corpus")
        run_id = create_sync_run(connection)
        cursor.execute(
            "UPDATE hum_corpora SET run_id=%s WHERE id=%s AND user_id=%s",
            (run_id, payload["corpus_id"], user_id),
        )
        connection.commit()
        for external_id, track_id in _source_rows(user_id, credentials[0], payload["track_ids"]):
            check()
            try:
                cursor.execute(
                    "SELECT count(*) FROM melody_contours WHERE track_id=%s AND run_id=%s",
                    (track_id, run_id),
                )
                existing = cursor.fetchone()[0]
                if not existing:
                    report({"phase": "preprocess", "message": "Preparing shared melody audio"})
                    audio = client.audio_bytes(external_id)
                    prepare_audio(audio, mono_rates=(44100,), stereo=True, melody=True, check=check)
                    report({"phase": "melody-index", "message": "Extracting melody contours"})
                    summary["contours"] += store_track_contours(connection, track_id, run_id, audio)
                    del audio
                cursor.execute(
                    "INSERT INTO hum_corpus_tracks(corpus_id,track_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                    (payload["corpus_id"], track_id),
                )
                connection.commit()
                summary["tracks"] += 1
            except Exception:
                connection.rollback()
                summary["failed"] += 1
            report({"phase": "melody-index", "summary": dict(summary)})
    return summary


def execute(job: dict, context) -> dict | None:
    """Execute a leaf, or atomically expand a parent and return None.

    context.check must raise a BaseException-derived cancellation signal.
    jobs.expand owns parent waiting state and analysis_batch child creation.
    """
    context.check()
    if job["kind"] == "recording_search":
        from .recording_search import execute as search_recording

        return search_recording(job, context)
    if job["kind"] == "modal_setup":
        from .external_processing import execute_setup

        return execute_setup(job, context)
    kind = job["kind"]
    payload = dict(job.get("payload") or {})
    if "connection_id" not in payload and job.get("connection_id"):
        payload["connection_id"] = str(job["connection_id"])
    operation = payload.get("operation") if kind == "analysis_batch" else kind
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported analysis operation: {operation}")
    user_id = job["user_id"]

    def report(update):
        context.check()
        context.report(update)
        context.check()

    from . import main  # Transitional credential and ownership helpers only.

    credentials = None
    if operation not in {"audio_profiles", "semantic_fusion_build"}:
        credentials = main._load_connection(payload["connection_id"], user_id)
        if credentials is None:
            raise ValueError("Connection unavailable")
    # Where this job computes. Batches carry their sync's choice; a parent job uses
    # the sync's choice or, for other jobs, the instance default. Feature switches
    # and sync planning depend on it, so it is set before any planning.
    compute = "local"
    if operation not in {"audio_profiles", "semantic_fusion_build"}:
        if kind == "analysis_batch":
            compute = payload.get("compute") or "local"
        else:
            from .external_processing import compute_for

            compute = payload.get("compute") or compute_for(user_id, None)
    with (
        batch_audio_cache(context.check),
        preprocessing_session(context.check),
        ExitStack() as stack,
    ):
        stack.enter_context(remote_compute.computing_on(compute))
        if operation == "semantic_fusion_build":
            from .semantic_fusion import build_semantic_fusion

            report(
                {
                    "phase": "building",
                    "message": "Building semantic fusion vectors",
                    "completed": 0,
                    "total": 1,
                    "unit": "builds",
                }
            )
            return build_semantic_fusion(progress=report)
        if kind != "analysis_batch":
            size = int(get_settings().batch_size)
            if size <= 0:
                raise ValueError("ECHORA_BATCH_SIZE must be positive")
            ids = payload.get("track_ids")
            if operation == "navidrome_sync":
                report({"phase": "scanning", "message": "Scanning complete catalog"})
                with NavidromeClient(*credentials) as client:
                    catalog = client.all_tracks()
                    catalog_ids = [track.id for track in catalog]
                context.check()
                from .sync_plan import select_sync_tracks

                report(
                    {"phase": "planning", "message": "Checking missing analysis before batching"}
                )
                with _connect() as connection:
                    ids = select_sync_tracks(
                        connection,
                        credentials[0],
                        catalog_ids,
                        payload.get("mode", "all"),
                        catalog=catalog,
                        verify_audio_hashes=payload.get("verify_audio_hashes", True),
                    )
                context.check()
                main._attach_user_library(user_id, credentials[0])
                # Reconcile the complete snapshot, never only the work selection.
                main._reconcile_user_tracks(user_id, credentials[0], catalog_ids)
                report(
                    {
                        "phase": "planning",
                        "message": f"{len(ids)} songs need processing",
                        "total": len(ids),
                        "completed": 0,
                        "unit": "tracks",
                    }
                )
            elif operation == "import":
                if ids is None:
                    raise ValueError("import requires track_ids")
            elif operation == "audio_profiles":
                allowed = {str(v) for v in main._user_audio_track_ids(user_id)}
                ids = sorted(allowed) if ids is None else [str(v) for v in ids if str(v) in allowed]
            else:
                ids = [row[0] for row in _source_rows(user_id, credentials[0], ids)]
            ids = list(dict.fromkeys(str(v) for v in ids))
            if operation == "hum_corpus":
                from .hum_search import DEFAULT_CORPUS_SIZE

                limit = int(payload.get("track_limit", DEFAULT_CORPUS_SIZE))
                if not 1 <= limit <= 500:
                    raise ValueError("track_limit must be between 1 and 500")
                ids = ids[:limit]
                payload["corpus_id"] = str(job["id"])
                context.check()
                with _connect() as connection, connection.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO hum_corpora(id,user_id,status,track_limit)
                           VALUES (%s,%s,'building',%s) ON CONFLICT (id) DO NOTHING""",
                        (payload["corpus_id"], user_id, limit),
                    )
                    cursor.execute(
                        "SELECT user_id FROM hum_corpora WHERE id=%s", (payload["corpus_id"],)
                    )
                    if str(cursor.fetchone()[0]) != str(user_id):
                        raise ValueError("Hum corpus ownership mismatch")
            # Whitelist fields: never forward arbitrary request data into the queue.
            base = {"operation": operation}
            if operation == "navidrome_sync":
                base["verify_audio_hashes"] = payload.get("verify_audio_hashes", True)
                base["refresh_lyrics"] = payload.get("mode", "all") == "all"
            if credentials:
                base["connection_id"] = payload["connection_id"]
            if operation not in {"audio_profiles", "semantic_fusion_build"}:
                if compute == "modal":
                    from .external_processing import prepare

                    # Deploy and download once, before batches queue, with progress.
                    prepare(report, context.check).close()
                base["compute"] = compute
            if operation == "hum_corpus":
                base["corpus_id"] = payload["corpus_id"]
            batches = [{**base, "track_ids": ids[i : i + size]} for i in range(0, len(ids), size)]
            context.check()
            from . import jobs  # Worker/API integration may be installed independently.

            if jobs.expand(job["id"], job.get("claim_token") or context.token, batches) is False:
                raise jobs.JobCancelled()
            return None

        ids = payload["track_ids"]
        if not ids:
            return {"total": 0}
        if operation == "audio_profiles":
            allowed = {str(v) for v in main._user_audio_track_ids(user_id)}
            return _profiles([v for v in ids if str(v) in allowed], report)
        artwork_prompts = None
        if operation in {"navidrome_sync", "import"}:
            # Motion artwork prompts from External AI first, before any GPU stage starts, so
            # no GPU (here or on Modal) idles while they are written.
            from .motion_artwork_jobs import prepare_prompts

            try:
                artwork_prompts = prepare_prompts(
                    credentials, ids, progress=report, check=context.check
                )
            except Exception:
                # Optional: the render stage writes any prompt still missing.
                logger.exception("Could not prepare motion artwork prompts")
        # Model computation runs on Modal for this batch when the job chose it.
        # Each batch re-checks the deployment, so none runs stale code or models.
        from .external_processing import open_session

        stack.enter_context(
            remote_compute.session(compute, factory=lambda: open_session(report, context.check))
        )
        summary = {}
        if operation in {"navidrome_sync", "import"}:
            from .ingest import ingest_navidrome

            summary.update(
                asdict(
                    ingest_navidrome(
                        *credentials,
                        ids,
                        report,
                        verify_audio_hashes=payload.get("verify_audio_hashes", True),
                    )
                )
            )
            context.check()
            main._attach_user_library(user_id, credentials[0])
            summary["linked"] = _add_links(user_id, credentials[0], ids)
            canonical = list({row[1] for row in _source_rows(user_id, credentials[0], ids)})
            summary["audio_profiles"] = _profiles(canonical, report)
        if operation in {"navidrome_sync", "import", "lyrics_backfill", "karaoke_backfill"}:
            from .lyrics_pipeline import backfill_lyrics

            summary["lyrics"] = backfill_lyrics(
                *credentials,
                progress=report,
                external_ids=ids,
                only_missing=True,
                refresh_existing=payload.get("refresh_lyrics", False),
            )
        if operation in {"navidrome_sync", "import", "lyrics_backfill"}:
            from .translation_pipeline import backfill_translations

            summary["translations"] = backfill_translations(
                user_id, credentials[0], ids, check=context.check, progress=report
            )
        if operation in {"navidrome_sync", "import", "lyrics_backfill"}:
            from .lyric_themes import backfill_themes

            summary["themes"] = backfill_themes(
                user_id, credentials[0], ids, check=context.check, progress=report
            )
        if operation in {"navidrome_sync", "import", "lyrics_backfill", "karaoke_backfill"}:
            from .karaoke_pipeline import backfill_karaoke

            summary["karaoke"] = backfill_karaoke(*credentials, progress=report, external_ids=ids)
        if operation in {"navidrome_sync", "import", "voice_backfill"}:
            from .voice_pipeline import backfill_voice

            summary["voice"] = backfill_voice(*credentials, progress=report, external_ids=ids)
        if operation in {"navidrome_sync", "import"}:
            # Last, so earlier stages have released their models before ComfyUI needs the GPU.
            from .motion_artwork_jobs import render_batch

            artwork = render_batch(
                credentials,
                user_id,
                ids,
                progress=report,
                check=context.check,
                prompts=artwork_prompts,
            )
            if artwork is not None:
                summary["motion_artwork"] = artwork
        if operation == "recordings_backfill":
            summary = _recordings(credentials, user_id, ids, report, context.check)
        if operation == "hum_corpus":
            summary = _hum(credentials, user_id, payload, report, context.check)
        context.check()
        return summary
