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

import core.jobs.queue as jobs
from core.recordings import store
from worker.recordings.config import RecordingConfig
from worker.recordings.errors import QueueFull

MAX_CLIP_SECONDS = 20
SAMPLE_RATE = 8000


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
        count = store.indexed_song_count(db, config.representation_id, user_id)
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
    coverage = store.catalog_coverage(connection, config.representation_id, namespace, ids, user_id)
    indexed, unsupported = coverage["indexed"], coverage["unsupported"]
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
        store.expire_searches(db)


def enqueue(user_id, audio: bytes, representation_id: str, policy_id: str):
    if not audio or len(audio) > RecordingConfig.load().max_upload_bytes:
        raise ValueError("Invalid recording upload size")
    with jobs._db() as db:
        # Serialize admission, including the global cap, across API replicas.
        store.lock_admission(db)
        active = store.active_searches(db, user_id)
        recent = store.recent_searches(db, user_id)
        if (
            active["total"] >= RecordingConfig.load().max_active
            or active["owned"] >= RecordingConfig.load().max_active_per_user
            or recent >= RecordingConfig.load().uploads_per_minute
        ):
            raise QueueFull()
        public = jobs._insert(db, "recording_search", "analysis", user_id, None, {}, None, None)
        store.add_search(
            db,
            public["id"],
            representation_id,
            policy_id,
            audio,
            RecordingConfig.load().audio_retention_seconds,
            RecordingConfig.load().result_retention_seconds,
        )
        # Operator-armed only after explicit user consent. Consume the one-shot
        # reservation in the same transaction as admission, never on rejected
        # uploads. Later searches retain the usual terminal audio erasure.
        store.capture_diagnostic(
            db, audio, public["id"], RecordingConfig.load().diagnostic_retention_seconds, user_id
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


def needs_fingerprints(connection, track_id, audio: bytes, config) -> bool:
    """Whether a song lacks recording fingerprints for this encoder.

    Raises ``ValueError`` when the song's bytes are not those its identity was resolved from.
    """
    if store.has_fingerprints(connection, track_id, config.representation_id):
        return False
    audio_hash = store.track_audio_hash(connection, track_id)
    if audio_hash is None or audio_hash != hashlib.sha256(audio).hexdigest():
        raise ValueError(
            "Recording source bytes changed; track identity must be refreshed before indexing"
        )
    return True


def encode_fingerprints(audio: bytes, config, check=lambda: None) -> np.ndarray:
    """The recording fingerprints of a song: one 128-dimensional vector per segment."""
    from worker.media.audio import decode_audio
    from worker.recordings.encoder import encode

    check()
    samples = decode_audio(audio, SAMPLE_RATE)
    fingerprints = (
        encode(samples, config, check=check)
        if len(samples) >= SAMPLE_RATE
        else np.empty((0, 128), dtype=np.float32)
    )
    check()
    return fingerprints


def insert_fingerprints(connection, track_id, fingerprints: np.ndarray, config) -> bool:
    """Store a song's recording fingerprints; ``False`` when another worker stored them first."""
    return store.save_fingerprints(
        connection,
        track_id,
        config.representation_id,
        len(fingerprints),
        np.asarray(fingerprints, dtype="<f4").tobytes(),
    )


def _references(db, user_id, representation_id, check):
    # Server-side cursor bounds memory. Access is scoped BEFORE similarity search.
    for row in store.reference_rows(db, representation_id, user_id):
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
        query = store.pending_search(db, job["id"], job["user_id"])
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
        if not store.lock_live_claim(db, job["id"], context.token):
            raise jobs.JobCancelled()
        store.save_search_result(db, job["id"], result_value)
    # Results live outside public job summaries, because access can later change.
    return {"state": result_value["state"]}


def _visible_matches(db, matches, user_id):
    if not matches:
        return []
    ids = [UUID(item["track_id"]) for item in matches]
    rows = store.visible_songs(db, user_id, ids)
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
        row = store.search_result(db, job_id, user_id)
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
