"""One-shot diagnostic retention never changes ordinary search retention."""

from uuid import uuid4

import core.jobs.queue as jobs
import pytest
import worker.recordings.search as search


def arm(database, owner, *, expired=False):
    with database() as db:
        db.execute("INSERT INTO users(id) VALUES (%s)", (owner,))
        db.execute(
            """INSERT INTO recording_diagnostic_captures(user_id,consent_version)
                   VALUES (%s,'preserve-next-recording-v1')""",
            (owner,),
        )
        if expired:
            db.execute(
                "UPDATE recording_diagnostic_captures SET expires_at=now()-interval '1 minute' WHERE user_id=%s",
                (owner,),
            )


def test_only_next_admitted_owner_recording_is_preserved(recording_db):
    owner, other = uuid4(), uuid4()
    arm(recording_db, owner)
    search.enqueue(other, b"other person's audio", "model", "policy")
    with recording_db() as db:
        assert (
            db.execute(
                "SELECT audio FROM recording_diagnostic_captures WHERE user_id=%s", (owner,)
            ).fetchone()["audio"]
            is None
        )
    first = search.enqueue(owner, b"consented diagnostic audio", "model", "policy")
    with pytest.raises(search.QueueFull):
        search.enqueue(owner, b"rejected upload", "model", "policy")
    # Cancellation still erases the normal search copy, not the consented copy.
    jobs.cancel(first["job_id"], owner)
    second = search.enqueue(owner, b"later audio", "model", "policy")
    jobs.cancel(second["job_id"], owner)
    with recording_db() as db:
        capture = db.execute(
            "SELECT * FROM recording_diagnostic_captures WHERE user_id=%s", (owner,)
        ).fetchone()
        assert bytes(capture["audio"]) == b"consented diagnostic audio"
        assert str(capture["job_id"]) == first["job_id"]
        assert (capture["expires_at"] - capture["captured_at"]).total_seconds() == 86400
        assert (
            db.execute(
                "SELECT audio FROM recording_searches WHERE job_id=%s", (first["job_id"],)
            ).fetchone()["audio"]
            is None
        )
        assert (
            db.execute(
                "SELECT audio FROM recording_searches WHERE job_id=%s", (second["job_id"],)
            ).fetchone()["audio"]
            is None
        )
    assert "consented diagnostic audio" not in str(jobs.get_job(first["job_id"], owner))
    assert "consented diagnostic audio" not in str(search.result(first["job_id"], owner))


def test_expired_arm_is_not_consumed_and_cleanup_erases_capture(recording_db):
    owner = uuid4()
    arm(recording_db, owner, expired=True)
    search.enqueue(owner, b"ordinary upload", "model", "policy")
    with recording_db() as db:
        assert (
            db.execute(
                "SELECT audio FROM recording_diagnostic_captures WHERE user_id=%s", (owner,)
            ).fetchone()["audio"]
            is None
        )
    search.cleanup()
    with recording_db() as db:
        assert (
            db.execute("SELECT count(*) AS n FROM recording_diagnostic_captures").fetchone()["n"]
            == 0
        )
    other = uuid4()
    arm(recording_db, other)
    search.enqueue(other, b"expires later", "model", "policy")
    with recording_db() as db:
        db.execute(
            "UPDATE recording_diagnostic_captures SET expires_at=now()-interval '1 minute' WHERE user_id=%s",
            (other,),
        )
    search.cleanup()
    with recording_db() as db:
        assert (
            db.execute("SELECT count(*) AS n FROM recording_diagnostic_captures").fetchone()["n"]
            == 0
        )


def test_rejected_upload_does_not_consume_arm(recording_db):
    owner = uuid4()
    arm(recording_db, owner)
    with pytest.raises(ValueError):
        search.enqueue(owner, b"", "model", "policy")
    with recording_db() as db:
        assert (
            db.execute(
                "SELECT captured_at FROM recording_diagnostic_captures WHERE user_id=%s", (owner,)
            ).fetchone()["captured_at"]
            is None
        )
