"""Durable curation scheduling and publication.

The session advisory lock is shared by workers and recipe mutation routes. It is
held across remote I/O (not merely the selecting transaction). Publication intent
outlives jobs: a fresh manual job cannot bypass an ambiguous playlist creation.
"""

from worker.curation.config import CurationConfig
from core.library.connections import load_connection
from core.integrations.navidrome import NavidromeClient
from core.curation.preview import CurationRecipe, preview_curation
from core.errors import ConflictError, NotFoundError
from core.db.config import require_database_url
from contextlib import contextmanager
import json
import uuid

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _connect():
    return psycopg.connect(require_database_url(), row_factory=dict_row)


@contextmanager
def locked(curation_id, *, wait=True):
    # Dedicated session; closing it also releases the lock after exceptions.
    with _connect() as connection:
        key = "curation:" + str(curation_id)
        if wait:
            connection.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (key,))
        else:
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired", (key,)
            ).fetchone()["acquired"]
            if not acquired:
                yield None
                return
        connection.commit()
        try:
            yield connection
        finally:
            connection.rollback()
            connection.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                ("curation:" + str(curation_id),),
            )
            connection.commit()


def enqueue(curation_id, user_id):
    import core.jobs.queue as jobs

    with locked(curation_id) as connection:
        curation = connection.execute(
            "SELECT * FROM curations WHERE id=%s AND user_id=%s", (curation_id, user_id)
        ).fetchone()
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
        connection.execute(
            """UPDATE curations SET status=CASE WHEN status='draft' THEN 'pending' ELSE status END,
            next_refresh_at=CASE WHEN refresh_enabled THEN now()+make_interval(hours => refresh_interval_hours) END,
            updated_at=now() WHERE id=%s""",
            (curation_id,),
        )
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

    with _connect() as connection:
        due = connection.execute(
            """SELECT id FROM curations
            WHERE status='draft' OR (refresh_enabled AND next_refresh_at <= now())
            ORDER BY next_refresh_at NULLS FIRST LIMIT %s""",
            (CurationConfig.load().due_limit,),
        ).fetchall()
    count = 0
    for item in due:
        with locked(item["id"], wait=False) as connection:
            if connection is None:
                continue
            curation = connection.execute(
                """SELECT * FROM curations WHERE id=%s
                AND (status='draft' OR (refresh_enabled AND next_refresh_at <= now()))""",
                (item["id"],),
            ).fetchone()
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
            connection.execute(
                """UPDATE curations SET
                status=CASE WHEN status='draft' THEN 'pending' ELSE status END,
                next_refresh_at=CASE WHEN refresh_enabled THEN now()+make_interval(hours => refresh_interval_hours) END,
                updated_at=now() WHERE id=%s""",
                (curation["id"],),
            )
            connection.commit()
            count += 1
    return count


def assert_mutable(connection, curation_id, *, deleting=False, delete_remote=True):
    """Do not let recipe changes invalidate a recoverable publication."""

    pending = connection.execute(
        "SELECT * FROM curation_publications WHERE curation_id=%s AND state <> 'complete'",
        (curation_id,),
    ).fetchone()
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
    existing = [
        row["track_id"]
        for row in connection.execute(
            """SELECT crt.track_id
        FROM curation_revision_tracks crt JOIN curation_revisions cr ON cr.id=crt.revision_id
        WHERE cr.curation_id=%s AND cr.revision_number=(SELECT max(revision_number)
            FROM curation_revisions WHERE curation_id=%s) ORDER BY crt.position LIMIT %s""",
            (curation_id, curation_id, curation["track_limit"]),
        ).fetchall()
    ]
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
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT id::text AS id, title, artist FROM tracks WHERE id=ANY(%s)""",
                ([value for value in waypoint_ids if value],),
            )
            by_id = {row["id"]: row for row in cursor.fetchall()}

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
    revision = connection.execute(
        "SELECT coalesce(max(revision_number), 0)+1 AS value FROM curation_revisions WHERE curation_id=%s",
        (curation_id,),
    ).fetchone()["value"]
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
        curation = connection.execute(
            "SELECT * FROM curations WHERE id=%s AND user_id=%s", (curation_id, job["user_id"])
        ).fetchone()
        if curation is None:
            return {"curation_id": str(curation_id), "deleted": True}
        try:
            publication = connection.execute(
                """SELECT * FROM curation_publications
                WHERE curation_id=%s AND (job_id=%s OR state <> 'complete')
                ORDER BY (job_id=%s) DESC LIMIT 1""",
                (curation_id, job["id"], job["id"]),
            ).fetchone()
            if publication and publication["state"] == "complete":
                return _summary(curation_id, publication)
            if publication is None:
                context.report({"phase": "selecting"})
                plan = _prepare(curation, job["user_id"], connection)
                context.check()
                publication = connection.execute(
                    """INSERT INTO curation_publications
                    (id, curation_id, job_id, state, plan, playlist_id)
                    VALUES (%s,%s,%s,'prepared',%s,%s) RETURNING *""",
                    (
                        uuid.uuid4(),
                        curation_id,
                        job["id"],
                        Jsonb(plan),
                        curation["navidrome_playlist_id"],
                    ),
                ).fetchone()
                connection.commit()
            if str(publication["job_id"]) != str(job["id"]):
                # Adopt a failed/cancelled job's pending intent. A crash after
                # completing this replacement job must replay its summary too.
                connection.execute(
                    "UPDATE curation_publications SET job_id=%s, updated_at=now() WHERE id=%s",
                    (job["id"], publication["id"]),
                )
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
            connection.execute(
                "UPDATE curation_publications SET state='publishing', updated_at=now() WHERE id=%s",
                (publication["id"],),
            )
            # A killed subprocess leaves an honest recoverable state, never a
            # permanent 'refreshing' flag. The job is the source of live progress.
            connection.execute(
                "UPDATE curations SET status='pending', last_error='Publication pending; refresh resumes the saved selection', updated_at=now() WHERE id=%s",
                (curation_id,),
            )
            connection.commit()
            with NavidromeClient(*credentials) as client:
                playlist_id = client.replace_playlist(
                    plan["name"], plan["source_ids"], publication["playlist_id"]
                )
            # Store the returned ID before any potentially failing local revision
            # writes or cancellation checks. This shrinks the ambiguous-create gap.
            connection.execute(
                "UPDATE curation_publications SET playlist_id=%s, updated_at=now() WHERE id=%s",
                (playlist_id, publication["id"]),
            )
            connection.execute(
                "UPDATE curations SET navidrome_playlist_id=%s WHERE id=%s",
                (playlist_id, curation_id),
            )
            connection.commit()
            revision = connection.execute(
                """INSERT INTO curation_revisions (curation_id, revision_number, recipe)
                VALUES (%s,%s,%s) RETURNING id""",
                (curation_id, plan["revision_number"], Jsonb(plan["recipe"])),
            ).fetchone()["id"]
            for position, track in enumerate(plan["tracks"]):
                connection.execute(
                    """INSERT INTO curation_revision_tracks
                    (revision_id, position, track_id, score, evidence, source_id)
                    VALUES (%s,%s,%s,%s,%s,%s)""",
                    (
                        revision,
                        position,
                        track["id"],
                        track["score"],
                        Jsonb(
                            {
                                "percentile": track["percentile"],
                                "retained": track["retained"],
                                **track.get("evidence", {}),
                            }
                        ),
                        track["source_id"],
                    ),
                )
            connection.execute(
                """UPDATE curations SET status='ready', last_error=NULL,
                last_refreshed_at=now(), next_refresh_at=CASE WHEN refresh_enabled THEN now()+make_interval(hours => refresh_interval_hours) END,
                updated_at=now() WHERE id=%s""",
                (curation_id,),
            )
            connection.execute(
                "UPDATE curation_publications SET state='complete', updated_at=now() WHERE id=%s",
                (publication["id"],),
            )
            connection.commit()
            publication["playlist_id"] = playlist_id
            return _summary(curation_id, publication)
        except BaseException as error:
            connection.rollback()
            connection.execute(
                "UPDATE curations SET status='failed', last_error=%s, updated_at=now() WHERE id=%s",
                (str(error), curation_id),
            )
            connection.commit()
            raise


def _summary(curation_id, publication):
    return {
        "curation_id": str(curation_id),
        "playlist_id": publication["playlist_id"],
        "revision_number": publication["plan"]["revision_number"],
        "track_count": len(publication["plan"]["source_ids"]),
    }
