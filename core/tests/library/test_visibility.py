"""PostgreSQL coverage: both projection uniqueness constraints are intentional."""

import runpy
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from psycopg.rows import tuple_row

from core.library.visibility import lock_library, source_remapped, update_memberships


def setup(db):
    db.row_factory = tuple_row
    db.execute("""CREATE TABLE users(id uuid PRIMARY KEY);
        CREATE TABLE libraries(id uuid PRIMARY KEY);
        CREATE TABLE track_sources(library_id uuid,track_id uuid,source_type text,external_id text,
            UNIQUE(library_id,source_type,external_id));
        CREATE TABLE user_track_links(user_id uuid,library_id uuid,track_id uuid,external_id text,
            UNIQUE(user_id,library_id,track_id),UNIQUE(user_id,library_id,external_id));""")
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "src/db"
            / "migrations/versions/0047_source_membership.py"
        )
    )


def test_migration_backfills_only_proven_links(database):
    with database() as db:
        migration = setup(db)
        user, library, track = uuid4(), uuid4(), uuid4()
        db.execute("INSERT INTO users VALUES (%s)", (user,))
        db.execute("INSERT INTO libraries VALUES (%s)", (library,))
        db.execute(
            "INSERT INTO track_sources VALUES (%s,%s,'subsonic','proven'),(%s,%s,'subsonic','unproven')",
            (library, track, library, track),
        )
        db.execute(
            "INSERT INTO user_track_links VALUES (%s,%s,%s,'proven')", (user, library, track)
        )
        with patch("alembic.op.execute", side_effect=db.execute):
            migration["upgrade"]()
        assert db.execute("SELECT external_id FROM user_source_memberships").fetchall() == [
            ("proven",)
        ]


def test_snapshots_pending_identity_batches_and_isolation(database):
    with database() as db:
        migration = setup(db)
        with patch("alembic.op.execute", side_effect=db.execute):
            migration["upgrade"]()
        user, other, library, elsewhere, a, b = [uuid4() for _ in range(6)]
        db.execute("INSERT INTO users VALUES (%s),(%s)", (user, other))
        db.execute("INSERT INTO libraries VALUES (%s),(%s)", (library, elsewhere))
        db.execute(
            "INSERT INTO track_sources VALUES (%s,%s,'subsonic','one'),(%s,%s,'subsonic','unproven'),(%s,%s,'subsonic','one')",
            (library, a, library, a, elsewhere, a),
        )
        update_memberships(db, library, other, ["one"], full=True)
        update_memberships(db, elsewhere, user, ["one"], full=True)
        assert update_memberships(db, library, user, ["one", "pending"], full=True)["linked"] == 1
        # A batch must retain pending IDs from its sibling, even before identity exists.
        update_memberships(db, library, user, ["one"])
        lock_library(db, library)
        db.execute("INSERT INTO track_sources VALUES (%s,%s,'subsonic','pending')", (library, b))
        source_remapped(db, library, "pending")
        assert db.execute(
            "SELECT count(*) FROM user_track_links WHERE user_id=%s AND library_id=%s",
            (user, library),
        ).fetchone() == (2,)
        assert update_memberships(db, library, user, ["pending"], full=True) == {
            "linked": 1,
            "unlinked": 1,
        }
        assert set(db.execute("SELECT * FROM user_track_links").fetchall()) == {
            (user, library, b, "pending"),
            (other, library, a, "one"),
            (user, elsewhere, a, "one"),
        }
        assert db.execute(
            "SELECT external_id FROM user_source_memberships WHERE user_id=%s AND library_id=%s",
            (user, library),
        ).fetchall() == [("pending",)]
        assert update_memberships(db, library, user, [], full=True) == {"linked": 0, "unlinked": 1}
        assert db.execute(
            "SELECT count(*) FROM user_source_memberships WHERE external_id='unproven'"
        ).fetchone() == (0,)


def test_library_lock_is_transaction_scoped_and_library_local(database):
    library, elsewhere = uuid4(), uuid4()
    with database() as first, database() as second:
        second.row_factory = tuple_row
        lock_library(first, library)
        assert second.execute(
            "SELECT pg_try_advisory_xact_lock(hashtextextended(%s,0))",
            (f"source-visibility:{library}",),
        ).fetchone() == (False,)
        assert second.execute(
            "SELECT pg_try_advisory_xact_lock(hashtextextended(%s,0))",
            (f"source-visibility:{elsewhere}",),
        ).fetchone() == (True,)
        first.commit()
        assert second.execute(
            "SELECT pg_try_advisory_xact_lock(hashtextextended(%s,0))",
            (f"source-visibility:{library}",),
        ).fetchone() == (True,)
