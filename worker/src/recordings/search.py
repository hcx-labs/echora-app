"""Private recording queries and an exact, authorization-scoped retrieval baseline.

No model downloads or inference in the API. Fingerprint artifacts are immutable
per representation and become searchable in the same transaction as publication.
The exact reader deliberately precedes ANN so recall has a testable baseline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from uuid import UUID

import numpy as np
from psycopg.types.json import Jsonb

import core.jobs.queue as jobs
from worker.recordings.config import RecordingConfig

MAX_CLIP_SECONDS = 20
SAMPLE_RATE = 8000


class QueueFull(ValueError):
    pass


def search_configuration():
    from worker.recordings.encoder import config_from_env

    config = config_from_env()
    if config is None:
        raise ValueError("Recording encoder is disabled")
    path = RecordingConfig.load().match_policy
    if not path:
        raise ValueError("Recording match policy is not configured")
    policy, policy_id = load_match_policy(config, path)
    return config, policy, policy_id


def load_match_policy(config, path):
    """Validate a policy against a verified encoder without changing process env."""
    from worker.recordings.matcher import MATCHER_REVISION, MatchPolicy

    payload = Path(path).read_bytes()
    data = json.loads(payload)
    if (
        not isinstance(data, dict)
        or data.get("representation_id") != config.representation_id
        or data.get("matcher_revision") != MATCHER_REVISION
        or data.get("calibrated") is not True
        or not isinstance(data.get("validation_dataset"), str)
        or not data["validation_dataset"].strip()
        or not isinstance(data.get("thresholds"), dict)
    ):
        raise ValueError("A calibrated, representation-bound match policy is required")
    try:
        policy = MatchPolicy(**data["thresholds"])
    except TypeError as error:
        raise ValueError("Invalid recording match policy") from error
    return policy, hashlib.sha256(payload).hexdigest()


def status(user_id):
    from worker.recordings.calibration import enabled as calibration_enabled
    from worker.recordings.encoder import config_from_env

    calibration = calibration_enabled()
    unavailable = {
        "enabled": calibration,
        "recognition_enabled": False,
        "calibration_enabled": calibration,
    }
    try:
        config = config_from_env()
    except (ValueError, OSError):
        config = None
    if config is None:
        return {
            **unavailable,
            "indexed_tracks": 0,
            "reason": "Recording search needs a validated local model.",
        }
    with jobs._db() as db:
        count = db.execute(
            """SELECT count(*) AS n FROM recording_fingerprints f
            WHERE f.representation_id=%s AND f.segment_count>0 AND EXISTS (
                SELECT 1 FROM user_track_links u WHERE u.track_id=f.track_id AND u.user_id=%s)
            """,
            (config.representation_id, user_id),
        ).fetchone()["n"]
    try:
        search_configuration()
    except (ValueError, OSError):
        return {
            **unavailable,
            "indexed_tracks": count,
            "reason": "Recording search needs a calibrated recognition policy.",
        }
    return {
        "enabled": True,
        "recognition_enabled": True,
        "calibration_enabled": calibration,
        "indexed_tracks": count,
    }


def index_status(connection, namespace, external_ids, user_id):
    """Report the current model's coverage for this owner's scanned catalog.

    Count catalog IDs rather than global artifacts, including distinct source
    aliases of one recording. A zero-window artifact is terminal but unusable.
    This does not require, or enable, a recognition policy.
    """
    from worker.recordings.encoder import config_from_env

    ids = list(dict.fromkeys(external_ids))
    result = {
        "enabled": False,
        "total": len(ids),
        "indexed": 0,
        "unsupported": 0,
        "missing": len(ids),
    }
    try:
        config = config_from_env()
    except (ValueError, OSError):
        return {**result, "reason": "Recording model configuration needs attention."}
    if config is None:
        return {**result, "reason": "Recording fingerprinting needs a validated local model."}
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT count(DISTINCT ts.external_id) FILTER (WHERE f.segment_count>0),
                   count(DISTINCT ts.external_id) FILTER (WHERE f.segment_count=0)
            FROM track_sources ts JOIN libraries l ON l.id=ts.library_id
            JOIN recording_fingerprints f ON f.track_id=ts.track_id AND f.representation_id=%s
            WHERE l.namespace=%s AND ts.source_type='subsonic' AND ts.external_id=ANY(%s)
              AND EXISTS (SELECT 1 FROM user_track_links u WHERE u.user_id=%s
                  AND u.library_id=ts.library_id AND u.track_id=ts.track_id)""",
            (config.representation_id, namespace, ids, user_id),
        )
        indexed, unsupported = cursor.fetchone()
    return {
        **result,
        "enabled": True,
        "indexed": indexed,
        "unsupported": unsupported,
        "missing": len(ids) - indexed - unsupported,
    }


def cleanup():
    """Worker maintenance also expires abandoned uploads when no queries run."""
    with jobs._db() as db:
        db.execute(
            "UPDATE recording_searches SET audio=NULL WHERE audio IS NOT NULL AND audio_expires_at<=now()"
        )
        db.execute("DELETE FROM recording_searches WHERE expires_at<=now()")
        db.execute("DELETE FROM recording_diagnostic_captures WHERE expires_at<=now()")


def enqueue(user_id, audio: bytes, representation_id: str, policy_id: str):
    if not audio or len(audio) > RecordingConfig.load().max_upload_bytes:
        raise ValueError("Invalid recording upload size")
    with jobs._db() as db:
        # Serialize admission, including the global cap, across API replicas.
        db.execute("SELECT pg_advisory_xact_lock(hashtextextended('recording-search-admission',0))")
        active = db.execute(
            """SELECT count(*) AS total,
            count(*) FILTER (WHERE user_id=%s) AS owned FROM jobs
            WHERE kind='recording_search' AND status IN ('queued','running','waiting')""",
            (user_id,),
        ).fetchone()
        recent = db.execute(
            """SELECT count(*) AS n FROM jobs WHERE kind='recording_search'
            AND user_id=%s AND created_at>now()-interval '1 minute'""",
            (user_id,),
        ).fetchone()["n"]
        if (
            active["total"] >= RecordingConfig.load().max_active
            or active["owned"] >= RecordingConfig.load().max_active_per_user
            or recent >= RecordingConfig.load().uploads_per_minute
        ):
            raise QueueFull()
        public = jobs._insert(db, "recording_search", "analysis", user_id, None, {}, None, None)
        db.execute("UPDATE jobs SET max_attempts=1 WHERE id=%s", (public["id"],))
        db.execute(
            """INSERT INTO recording_searches(job_id,representation_id,policy_id,audio,audio_expires_at,expires_at)
            VALUES (%s,%s,%s,%s,now()+%s*interval '1 second',now()+%s*interval '1 second')""",
            (
                public["id"],
                representation_id,
                policy_id,
                audio,
                RecordingConfig.load().audio_retention_seconds,
                RecordingConfig.load().result_retention_seconds,
            ),
        )
        # Operator-armed only after explicit user consent. Consume the one-shot
        # reservation in the same transaction as admission, never on rejected
        # uploads. Later searches retain the usual terminal audio erasure.
        db.execute(
            """UPDATE recording_diagnostic_captures
            SET audio=%s,job_id=%s,captured_at=now(),expires_at=now()+%s*interval '1 second'
            WHERE user_id=%s AND captured_at IS NULL AND expires_at>now()""",
            (audio, public["id"], RecordingConfig.load().diagnostic_retention_seconds, user_id),
        )
        return {"job_id": public["id"]}


def decode_query(audio: bytes, check=lambda: None) -> np.ndarray:
    """Bound decoding and observe cancellation without retaining query PCM."""
    if not audio or len(audio) > RecordingConfig.load().max_upload_bytes:
        raise ValueError("Invalid recording upload size")
    check()
    deadline = time.monotonic() + RecordingConfig.load().decode_timeout_seconds
    with subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "pipe",
            "-threads",
            "1",
            "-i",
            "pipe:0",
            "-t",
            str(MAX_CLIP_SECONDS + 1),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as process:
        pending_input = audio
        try:
            while True:
                check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("Recording decode timed out")
                try:
                    output, _ = process.communicate(
                        input=pending_input,
                        timeout=min(RecordingConfig.load().cancel_poll_seconds, remaining),
                    )
                    break
                except subprocess.TimeoutExpired:
                    # communicate resumes partially written input and captured
                    # output on its next call. Do not resend the input bytes.
                    pending_input = None
        except BaseException:
            process.kill()
            process.communicate()
            raise
        check()
        if process.returncode or len(output) % 4:
            raise ValueError("Could not decode recording")
    samples = np.frombuffer(output, dtype="<f4").copy()
    # MediaRecorder's stop timer and codec padding can add a few frames. Allow
    # bounded capture jitter, then trim to the advertised search duration.
    if samples.size > (MAX_CLIP_SECONDS + 0.75) * SAMPLE_RATE:
        raise ValueError("Recordings must be at most 20 seconds")
    samples = samples[: MAX_CLIP_SECONDS * SAMPLE_RATE]
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Recording has no usable audio")
    return samples


def store_fingerprints(connection, track_id, audio: bytes, config, check=lambda: None):
    from worker.media.audio import decode_audio
    from worker.recordings.encoder import encode

    check()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM recording_fingerprints WHERE track_id=%s AND representation_id=%s",
            (track_id, config.representation_id),
        )
        if cursor.fetchone():
            return False
        cursor.execute("SELECT audio_hash FROM tracks WHERE id=%s", (track_id,))
        track = cursor.fetchone()
        if track is None or track[0] != hashlib.sha256(audio).hexdigest():
            raise ValueError(
                "Recording source bytes changed; track identity must be refreshed before indexing"
            )
    samples = decode_audio(audio, SAMPLE_RATE)
    fingerprints = (
        encode(samples, config, check=check)
        if len(samples) >= SAMPLE_RATE
        else np.empty((0, 128), dtype=np.float32)
    )
    check()
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO recording_fingerprints
            (track_id,representation_id,segment_count,fingerprints) VALUES (%s,%s,%s,%s)
            ON CONFLICT (track_id,representation_id) DO NOTHING""",
            (
                track_id,
                config.representation_id,
                len(fingerprints),
                np.asarray(fingerprints, dtype="<f4").tobytes(),
            ),
        )
        return cursor.rowcount == 1


def _references(db, user_id, representation_id, check):
    # Server-side cursor bounds memory. Access is scoped BEFORE similarity search.
    with db.cursor(name="recording_references") as cursor:
        cursor.itersize = 8
        cursor.execute(
            """SELECT f.track_id,f.segment_count,f.fingerprints,g.group_id
            FROM recording_fingerprints f
            LEFT JOIN recording_group_members g ON g.track_id=f.track_id
            WHERE f.representation_id=%s AND f.segment_count>0 AND EXISTS (
                SELECT 1 FROM user_track_links u WHERE u.track_id=f.track_id AND u.user_id=%s)
            ORDER BY f.track_id""",
            (representation_id, user_id),
        )
        for row in cursor:
            check()
            values = np.frombuffer(bytes(row["fingerprints"]), dtype="<f4")
            if values.size != row["segment_count"] * 128:
                raise ValueError("Corrupt recording fingerprint artifact")
            yield {
                "track_id": str(row["track_id"]),
                "recording_group_id": str(row["group_id"]) if row["group_id"] else None,
                "fingerprints": values.reshape(-1, 128),
            }


def execute(job, context):
    from worker.recordings.encoder import encode
    from worker.recordings.matcher import match_recording

    started = time.monotonic()
    deadline = started + RecordingConfig.load().processing_timeout_seconds
    last_lease_check = started
    context.check()

    def check():
        nonlocal last_lease_check
        now = time.monotonic()
        if now > deadline:
            raise ValueError("Recording search exceeded its time budget")
        if now - last_lease_check >= 1:
            context.check()
            last_lease_check = now

    config, policy, policy_id = search_configuration()
    check()
    with jobs._db() as db:
        query = db.execute(
            """SELECT q.* FROM recording_searches q JOIN jobs j ON j.id=q.job_id
            WHERE q.job_id=%s AND j.user_id=%s AND q.audio_expires_at>now()""",
            (job["id"], job["user_id"]),
        ).fetchone()
    if (
        not query
        or query["audio"] is None
        or query["representation_id"] != config.representation_id
        or query["policy_id"] != policy_id
    ):
        raise ValueError("Recording expired or recognition configuration changed")
    context.report({"phase": "recording_decode", "message": "Preparing recording"})
    samples = decode_query(bytes(query["audio"]), check=check)
    check()
    context.check()
    context.report({"phase": "recording_encode", "message": "Analyzing recording"})
    fingerprints = (
        encode(samples, config, check=check)
        if len(samples) >= SAMPLE_RATE
        else np.empty((0, 128), dtype=np.float32)
    )
    del samples, query
    check()
    context.report({"phase": "recording_match", "message": "Searching your indexed recordings"})
    with jobs._db() as db:
        result_value = match_recording(
            fingerprints,
            _references(db, job["user_id"], config.representation_id, check),
            policy,
            check=check,
        )
    context.check()
    with jobs._db() as db:
        # Lock the live job claim while publishing. A cancelled/expired executor
        # cannot overwrite a newer attempt's result or expose incomplete output.
        owned = db.execute(
            """SELECT id FROM jobs WHERE id=%s AND token=%s
            AND status='running' AND NOT cancel_requested AND lease_until>now()
            FOR UPDATE""",
            (job["id"], context.token),
        ).fetchone()
        if not owned:
            raise jobs.JobCancelled()
        db.execute(
            "UPDATE recording_searches SET result=%s WHERE job_id=%s",
            (Jsonb(result_value), job["id"]),
        )
    # Results live outside public job summaries, because access can later change.
    return {"state": result_value["state"]}


def _visible_matches(db, matches, user_id):
    if not matches:
        return []
    ids = [UUID(item["track_id"]) for item in matches]
    rows = db.execute(
        """SELECT t.id,t.title,t.artist,t.album,t.duration_seconds,
        CASE WHEN u.connection_id IS NOT NULL THEN u.external_id END AS source_id,
        u.connection_id::text AS connection_id,
        CASE WHEN u.connection_id IS NOT NULL THEN ts.source_data->>'coverArt' END AS cover_art
        FROM tracks t JOIN LATERAL (
            SELECT link.external_id,link.library_id,nc.id AS connection_id
            FROM user_track_links link
            LEFT JOIN libraries library ON library.id=link.library_id
            LEFT JOIN navidrome_connections nc ON nc.owner_user_id=link.user_id
                AND rtrim(nc.url,'/')=rtrim(library.root_path,'/')
            LEFT JOIN user_preferences preference ON preference.user_id=link.user_id
            WHERE link.track_id=t.id AND link.user_id=%s
            ORDER BY (nc.id=preference.navidrome_connection_id) DESC NULLS LAST,
                     nc.id NULLS LAST,link.library_id,link.external_id LIMIT 1
        ) u ON true
        LEFT JOIN track_sources ts ON ts.track_id=t.id AND ts.library_id=u.library_id
            AND ts.external_id=u.external_id AND ts.source_type='subsonic'
        WHERE t.id=ANY(%s)""",
        (user_id, ids),
    ).fetchall()
    visible = {str(row["id"]): row for row in rows}
    return [
        {
            **{key: value for key, value in match.items() if key != "recording_group_id"},
            **{key: value for key, value in visible[match["track_id"]].items() if key != "id"},
        }
        for match in matches
        if match["track_id"] in visible
    ]


def result(job_id, user_id):
    with jobs._db() as db:
        row = db.execute(
            """SELECT j.status,j.error,q.result FROM jobs j
            LEFT JOIN recording_searches q ON q.job_id=j.id AND q.expires_at>now()
            WHERE j.id=%s AND j.user_id=%s AND j.kind='recording_search'""",
            (job_id, user_id),
        ).fetchone()
        if not row:
            return None
        response = {"status": row["status"]}
        if row["status"] == "failed":
            response["error"] = "Recording search failed. Check the clip length and try again."
        if row["status"] == "complete":
            if row["result"] is None:
                return {"status": "failed", "error": "Recording search results expired."}
            value = row["result"]
            visible = _visible_matches(db, value.get("matches", []), user_id)
            state = value["state"]
            # Never promote an old runner-up after the identified track is revoked.
            if value.get("matches") and (
                not visible or visible[0]["track_id"] != value["matches"][0]["track_id"]
            ):
                state = "no_match"
                visible = []
            response["result"] = {"state": state, "matches": visible}
        return response
