"""Analysis operations: which stages each runs, and how its job becomes batches.

Queue payloads contain references, never secrets. A parent job carries ``connection_id`` and
optional external ``track_ids`` (import requires them); audio profiles take canonical track UUIDs
and no connection. A parent plans its songs and expands into ``analysis_batch`` children with the
same whitelisted payload plus ``operation`` and a bounded ``track_ids``. A hum corpus parent
creates the corpus under its own job id and passes ``corpus_id`` to its batches. User identity
belongs to the job envelope, not the payload.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import worker.remote.session as remote_compute
from core.compute import computing_on
from core.integrations.navidrome import NavidromeClient, batch_audio_cache
from core.library.connections import load_connection
from core.library.membership import (
    attach_user_library,
    reconcile_user_tracks,
    relink_renumbered_sources,
    user_audio_track_ids,
)
from worker.config import WorkerConfig
from worker.media.preprocessing import preprocessing_session
from worker.stages import (
    AUDIO_PROFILES,
    KARAOKE,
    LIBRARY,
    LYRICS,
    MELODY,
    MOTION_ARTWORK,
    RECORDINGS,
    SEMANTIC_FUSION,
    THEMES,
    TRANSLATIONS,
    VOICE,
    Batch,
    Stage,
    connect,
    source_rows,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Operation:
    """What one kind of analysis job does."""

    stages: tuple[Stage, ...]
    library: bool = True
    """Reads songs from a Navidrome connection (and may compute on Modal)."""
    batched: bool = True
    """A parent job expands into batches; otherwise the parent runs the stages itself."""


SONG_STAGES = (LIBRARY, LYRICS, TRANSLATIONS, THEMES, KARAOKE, VOICE, MOTION_ARTWORK)

OPERATIONS: dict[str, Operation] = {
    "navidrome_sync": Operation(SONG_STAGES),
    "import": Operation(SONG_STAGES),
    "lyrics_backfill": Operation((LYRICS, TRANSLATIONS, THEMES, KARAOKE)),
    # Karaoke aligns the lyrics the lyrics stage keeps current.
    "karaoke_backfill": Operation((LYRICS, KARAOKE)),
    "voice_backfill": Operation((VOICE,)),
    "recordings_backfill": Operation((RECORDINGS,)),
    "hum_corpus": Operation((MELODY,)),
    "audio_profiles": Operation((AUDIO_PROFILES,), library=False),
    "semantic_fusion_build": Operation((SEMANTIC_FUSION,), library=False, batched=False),
}


def execute(job: dict, context) -> dict | None:
    """Run a batch (or an unbatched operation) and return its summary, or expand a parent.

    ``context.check`` must raise a ``BaseException`` cancellation signal. ``jobs.expand`` owns the
    parent's waiting state and the creation of its batches; expanding returns ``None``.
    """
    context.check()
    kind = job["kind"]
    if kind == "recording_search":
        from worker.recordings.search import execute as search_recording

        return search_recording(job, context)
    if kind == "modal_setup":
        from worker.remote.settings import execute_setup

        return execute_setup(job, context)
    payload = dict(job.get("payload") or {})
    if "connection_id" not in payload and job.get("connection_id"):
        payload["connection_id"] = str(job["connection_id"])
    name = payload.get("operation") if kind == "analysis_batch" else kind
    operation = OPERATIONS.get(name)
    if operation is None:
        raise ValueError(f"Unsupported analysis operation: {name}")
    user_id = job["user_id"]

    def report(update):
        context.check()
        context.report(update)
        context.check()

    credentials = None
    compute = "local"
    if operation.library:
        credentials = load_connection(payload["connection_id"], user_id)
        if credentials is None:
            raise ValueError("Connection unavailable")
        # Where this job computes. Batches carry their sync's choice; a parent uses the request's
        # or the instance default. Feature switches and sync planning depend on it.
        compute = payload.get("compute") or "local"
        if kind != "analysis_batch" and not payload.get("compute"):
            from worker.remote.settings import compute_for

            compute = compute_for(user_id, None)
    batch = Batch(
        user_id=user_id,
        track_ids=list(payload.get("track_ids") or []),
        payload=payload,
        report=report,
        check=context.check,
        credentials=credentials,
    )
    with (
        batch_audio_cache(context.check),
        preprocessing_session(context.check),
        ExitStack() as stack,
    ):
        stack.enter_context(computing_on(compute))
        if kind != "analysis_batch" and operation.batched:
            _expand(job, context, name, operation, batch, compute)
            return None
        if operation.batched and not batch.track_ids:
            return {"total": 0}
        return _run(operation, batch, compute, stack, context)


def _run(operation: Operation, batch: Batch, compute: str, stack: ExitStack, context) -> dict:
    for stage in operation.stages:
        if stage.prepare is not None:
            try:
                batch.prepared[stage.key] = stage.prepare(batch)
            except Exception as error:  # noqa: BLE001 - preparation is optional
                # Optional: the stage itself does whatever preparation is still missing.
                logger.error("Could not prepare %s: %s", stage.key, type(error).__name__)
    if operation.library:
        # Model computation runs on Modal for this batch when the job chose it. Each batch
        # re-checks the deployment, so none runs stale code or models.
        from worker.remote.settings import open_session

        stack.enter_context(
            remote_compute.session(
                compute, factory=lambda: open_session(batch.report, context.check)
            )
        )
    summary: dict[str, Any] = {}
    for stage in operation.stages:
        summary.update(stage.run(batch))
    context.check()
    return summary


def _expand(job: dict, context, name: str, operation: Operation, batch: Batch, compute: str):
    """Plan the parent's songs and queue them as batches."""
    payload, user_id = batch.payload, batch.user_id
    ids = payload.get("track_ids")
    size = WorkerConfig.load().batch_size
    if name == "navidrome_sync":
        ids = _plan_sync(batch)
    elif name == "import":
        if ids is None:
            raise ValueError("import requires track_ids")
    elif name == "audio_profiles":
        allowed = {str(v) for v in user_audio_track_ids(user_id)}
        ids = sorted(allowed) if ids is None else [str(v) for v in ids if str(v) in allowed]
    else:
        ids = [row[0] for row in source_rows(user_id, batch.url, ids)]
    ids = list(dict.fromkeys(str(v) for v in ids))
    if name == "hum_corpus":
        ids = _create_corpus(job, context, batch, ids)
    # Whitelist fields: never forward arbitrary request data into the queue.
    base: dict[str, Any] = {"operation": name}
    if name == "navidrome_sync":
        base["verify_audio_hashes"] = payload.get("verify_audio_hashes", True)
        base["refresh_lyrics"] = payload.get("mode", "all") == "all"
    if operation.library:
        base["connection_id"] = payload["connection_id"]
        if compute == "modal":
            from worker.remote.settings import prepare

            # Deploy and download once, before batches queue, with progress.
            prepare(batch.report, context.check).close()
        base["compute"] = compute
    if name == "hum_corpus":
        base["corpus_id"] = payload["corpus_id"]
    batches = [{**base, "track_ids": ids[i : i + size]} for i in range(0, len(ids), size)]
    context.check()
    import core.jobs.queue as jobs

    if jobs.expand(job["id"], job.get("claim_token") or context.token, batches) is False:
        raise jobs.JobCancelled()


def _plan_sync(batch: Batch) -> list[str]:
    """The songs of a full scan that need processing; reconciles the user's library first."""
    from worker.planning.sync import select_sync_tracks

    payload = batch.payload
    batch.report({"phase": "scanning", "message": "Scanning complete catalog"})
    with NavidromeClient(*batch.library) as client:
        catalog = client.all_tracks()
        catalog_ids = [track.id for track in catalog]
    batch.check()
    batch.report({"phase": "planning", "message": "Checking missing analysis before batching"})
    with connect() as connection:
        # A renumbered song keeps its analysis; it is not new work.
        relink_renumbered_sources(connection, batch.url, catalog)
        ids = select_sync_tracks(
            connection,
            batch.url,
            catalog_ids,
            payload.get("mode", "all"),
            catalog=catalog,
            verify_audio_hashes=payload.get("verify_audio_hashes", True),
        )
    batch.check()
    attach_user_library(batch.user_id, batch.url)
    # Reconcile the complete snapshot, never only the work selection.
    reconcile_user_tracks(batch.user_id, batch.url, catalog_ids)
    batch.report(
        {
            "phase": "planning",
            "message": f"{len(ids)} songs need processing",
            "total": len(ids),
            "completed": 0,
            "unit": "tracks",
        }
    )
    return ids


def _create_corpus(job: dict, context, batch: Batch, ids: list[str]) -> list[str]:
    """Create the hum corpus under the parent's id; returns the songs it will index."""
    from worker.melody.index import DEFAULT_CORPUS_SIZE

    limit = int(batch.payload.get("track_limit", DEFAULT_CORPUS_SIZE))
    if not 1 <= limit <= 500:
        raise ValueError("track_limit must be between 1 and 500")
    batch.payload["corpus_id"] = str(job["id"])
    context.check()
    from core.melody import store

    with connect() as connection:
        store.add_corpus(connection, batch.payload["corpus_id"], batch.user_id, limit)
        if str(store.corpus_owner(connection, batch.payload["corpus_id"])) != str(batch.user_id):
            raise ValueError("Hum corpus ownership mismatch")
    return ids[:limit]
