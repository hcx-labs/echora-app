"""Durable curation scheduling and publication.

The session advisory lock is shared by workers and recipe mutation routes. It is
held across remote I/O (not merely the selecting transaction). Publication intent
outlives jobs: a fresh manual job cannot bypass an ambiguous playlist creation.
"""

import json
import uuid
from contextlib import contextmanager

from core.curation import store
from core.curation.preview import CurationRecipe, preview_curation
from core.db.connection import connect
from core.errors import ConflictError, NotFoundError
from core.integrations.navidrome import NavidromeClient
from core.library.connections import load_connection
from worker.curation.config import CurationConfig


@contextmanager
def locked(curation_id, *, wait=True):
    # Dedicated session; closing it also releases the lock after exceptions.
    with connect(dict_rows=True) as connection:
        if wait:
            store.lock_curation(connection, curation_id)
        elif not store.try_lock_curation(connection, curation_id):
            yield None
            return
        connection.commit()
        try:
            yield connection
        finally:
            connection.rollback()
            store.unlock_curation(connection, curation_id)
            connection.commit()


def enqueue(curation_id, user_id):
    import core.jobs.queue as jobs

    with locked(curation_id) as connection:
        curation = store.curation_row(connection, curation_id, user_id)
        if curation is None:
            raise NotFoundError("Curation not found")
        job = jobs.enqueue(
            kind="curation_refresh",
            worker_type="scheduled",
            user_id=user_id,
            connection_id=curation["navidrome_connection_id"],
            payload={"curation_id": str(curation_id)},
            dedupe_key="curation:" + str(curation_id),
        )
        store.mark_queued(connection, curation_id)
        connection.commit()
        return {
            "job_id": str(job.get("job_id") or job["id"]),
            "status": job["status"],
            "curation_id": str(curation_id),
        }


def enqueue_due():
    """Safe concurrent polling; advance due time only after durable enqueue.

    Drafts cover a crash between committing creation and enqueue, even when
    automatic refresh is disabled. The active-job unique key closes the crash
    window between enqueue and advancing the schedule.
    """
    import core.jobs.queue as jobs

    with connect() as connection:
        due = store.due_curation_ids(connection, CurationConfig.load().due_limit)
    count = 0
    for curation_id in due:
        with locked(curation_id, wait=False) as connection:
            if connection is None:
                continue
            curation = store.due_curation(connection, curation_id)
            if curation is None:
                continue
            jobs.enqueue(
                kind="curation_refresh",
                worker_type="scheduled",
                user_id=curation["user_id"],
                connection_id=curation["navidrome_connection_id"],
                payload={"curation_id": str(curation["id"])},
                dedupe_key="curation:" + str(curation["id"]),
            )
            store.mark_queued(connection, curation["id"])
            connection.commit()
            count += 1
    return count


def assert_mutable(connection, curation_id, *, deleting=False, delete_remote=True):
    """Do not let recipe changes invalidate a recoverable publication."""
    pending = store.unresolved_publication(connection, curation_id)
    if pending and not deleting:
        raise ConflictError("A publication is unresolved; refresh it before changing the recipe")
    if (
        pending
        and deleting
        and delete_remote
        and pending["state"] == "publishing"
        and not pending["playlist_id"]
    ):
        raise ConflictError(
            "Playlist creation outcome is unknown; reconcile Navidrome or explicitly keep the remote playlist"
        )


def _json(value):
    return json.loads(json.dumps(value, default=str))


def _prepare(curation, user_id, connection):

    curation_id = curation["id"]
    existing = store.latest_revision_track_ids(connection, curation_id, curation["track_limit"])
    request = CurationRecipe(
        curation_type=curation["curation_type"],
        positive_prompt=curation["positive_prompt"],
        negative_prompt=curation["negative_prompt"],
        sound_prompts=curation.get("sound_prompts") or [],
        themes_prompts=curation.get("themes_prompts") or [],
        sound_negative_prompts=curation.get("sound_negative_prompts") or [],
        themes_negative_prompts=curation.get("themes_negative_prompts") or [],
        sound_weight=int(curation.get("sound_weight") or 50),
        sound_profile=curation.get("sound_profile") or {},
        positive_track_ids=curation["positive_track_ids"],
        negative_track_ids=curation["negative_track_ids"],
        familiarity_percent=curation["familiarity_percent"],
        period_start=curation["period_start"],
        period_end=curation["period_end"],
        lookback_days=curation["lookback_days"],
        time_of_day_enabled=bool(curation.get("time_of_day_enabled")),
        track_limit=curation["track_limit"],
        refresh_mode=curation["refresh_mode"],
        target_language=str(curation.get("target_language") or ""),
        language_strictness=str(curation.get("language_strictness") or "primarily"),
        existing_track_ids=existing,
        journey_start_track_id=curation.get("journey_start_track_id"),
        journey_stop_track_ids=[
            uuid.UUID(str(value)) for value in (curation.get("journey_stop_track_ids") or [])
        ],
        journey_end_track_id=curation.get("journey_end_track_id"),
        journey_lyrics_weight=int(
            curation.get("journey_lyrics_weight")
            if curation.get("journey_lyrics_weight") is not None
            else 25
        ),
    )
    result = preview_curation(user_id, curation["navidrome_connection_id"], request)
    if not result["tracks"]:
        raise ValueError(
            "No tracks meet this recipe's requirements. Existing playlist was not changed."
        )
    recipe = {
        "curation_type": request.curation_type,
        "positive_prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "sound_prompts": request.sound_prompts,
        "themes_prompts": request.themes_prompts,
        "sound_negative_prompts": request.sound_negative_prompts,
        "themes_negative_prompts": request.themes_negative_prompts,
        "sound_weight": request.sound_weight,
        "sound_profile": request.sound_profile.model_dump(exclude_none=True, exclude_defaults=True),
        "positive_track_ids": [str(value) for value in request.positive_track_ids],
        "negative_track_ids": [str(value) for value in request.negative_track_ids],
        "familiarity_percent": request.familiarity_percent,
        "period_start": request.period_start,
        "period_end": request.period_end,
        "time_of_day_enabled": request.time_of_day_enabled,
        "lookback_days": request.lookback_days,
        "track_limit": request.track_limit,
        "refresh_mode": request.refresh_mode,
        "target_language": request.target_language,
        "language_strictness": request.language_strictness,
        **(
            {"journey_start": None, "journey_stops": [], "journey_end": None}
            if request.curation_type == "sonic_journey"
            else {}
        ),
        "journey_start_track_id": str(request.journey_start_track_id)
        if request.journey_start_track_id
        else None,
        "journey_stop_track_ids": [str(value) for value in request.journey_stop_track_ids],
        "journey_end_track_id": str(request.journey_end_track_id)
        if request.journey_end_track_id
        else None,
        "journey_lyrics_weight": request.journey_lyrics_weight,
        "references": result["references"],
        "model": result["model"],
        "weights": result.get("weights"),
        "lyrics_coverage": result.get("lyrics_coverage"),
        "signal_weights": result.get("signal_weights"),
        "scoring_revision": result.get("scoring_revision"),
        "selection": result.get("selection"),
        "representation_runs": result.get("representation_runs"),
        "example_component_weights": result.get("example_component_weights"),
        "audio_profile_coverage": result.get("audio_profile_coverage"),
        "familiarity": result.get("familiarity"),
        "shuffle_seed": result.get("shuffle_seed"),
    }
    if request.curation_type == "sonic_journey":
        waypoint_ids = [
            request.journey_start_track_id,
            *request.journey_stop_track_ids,
            request.journey_end_track_id,
        ]
        by_id = {
            row["id"]: row
            for row in store.track_names(connection, [value for value in waypoint_ids if value])
        }

        def reference(value):
            row = by_id.get(str(value))
            return (
                {"id": str(value), "title": row["title"], "artist": row.get("artist")}
                if row
                else None
            )

        recipe["journey_start"] = reference(request.journey_start_track_id)
        recipe["journey_stops"] = [reference(value) for value in request.journey_stop_track_ids]
        recipe["journey_end"] = reference(request.journey_end_track_id)
    revision = store.next_revision_number(connection, curation_id)
    return _json(
        {
            "name": curation["name"],
            "source_ids": [str(t["source_id"]) for t in result["tracks"]],
            "tracks": result["tracks"],
            "recipe": recipe,
            "revision_number": revision,
        }
    )


def execute(job, context):
    """Execute (or resume) the exact durable publication; return a job summary."""
    curation_id = uuid.UUID(str(job["payload"]["curation_id"]))
    with locked(curation_id) as connection:
        context.check()
        curation = store.curation_row(connection, curation_id, job["user_id"])
        if curation is None:
            return {"curation_id": str(curation_id), "deleted": True}
        try:
            publication = store.job_publication(connection, curation_id, job["id"])
            if publication and publication["state"] == "complete":
                return _summary(curation_id, publication)
            if publication is None:
                context.report({"phase": "selecting"})
                plan = _prepare(curation, job["user_id"], connection)
                context.check()
                publication = store.add_publication(
                    connection, curation_id, job["id"], plan, curation["navidrome_playlist_id"]
                )
                connection.commit()
            if str(publication["job_id"]) != str(job["id"]):
                # Adopt a failed/cancelled job's pending intent. A crash after
                # completing this replacement job must replay its summary too.
                store.adopt_publication(connection, publication["id"], job["id"])
                connection.commit()
            plan = publication["plan"]
            # Never repeat an initial create after an uncertain network outcome.
            # A known playlist ID makes exact replacement safely repeatable.
            if publication["state"] == "publishing" and not publication["playlist_id"]:
                raise RuntimeError(
                    "Navidrome playlist creation outcome is unknown; reconcile the remote playlist before retrying (automatic creation blocked)"
                )
            credentials = load_connection(str(curation["navidrome_connection_id"]))
            if credentials is None:
                raise RuntimeError("Navidrome connection is unavailable")
            context.report({"phase": "publishing", "track_count": len(plan["source_ids"])})
            context.check()
            # A killed subprocess leaves an honest recoverable state, never a
            # permanent 'refreshing' flag. The job is the source of live progress.
            store.mark_publishing(connection, publication["id"], curation_id)
            connection.commit()
            with NavidromeClient(*credentials) as client:
                playlist_id = client.replace_playlist(
                    plan["name"], plan["source_ids"], publication["playlist_id"]
                )
            # Store the returned ID before any potentially failing local revision
            # writes or cancellation checks. This shrinks the ambiguous-create gap.
            store.record_playlist(connection, publication["id"], curation_id, playlist_id)
            connection.commit()
            store.add_revision(
                connection,
                curation_id,
                plan["revision_number"],
                plan["recipe"],
                [
                    {
                        **track,
                        "evidence": {
                            "percentile": track["percentile"],
                            "retained": track["retained"],
                            **track.get("evidence", {}),
                        },
                    }
                    for track in plan["tracks"]
                ],
            )
            store.complete_publication(connection, publication["id"], curation_id)
            connection.commit()
            publication["playlist_id"] = playlist_id
            return _summary(curation_id, publication)
        except BaseException as error:
            connection.rollback()
            store.mark_failed(connection, curation_id, str(error))
            connection.commit()
            raise


def _summary(curation_id, publication):
    return {
        "curation_id": str(curation_id),
        "playlist_id": publication["playlist_id"],
        "revision_number": publication["plan"]["revision_number"],
        "track_count": len(publication["plan"]["source_ids"]),
    }
