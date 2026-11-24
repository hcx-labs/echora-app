"""Authorized source IDs are durable; canonical track links are only a projection.

Callers own the transaction. All source remaps and membership writes take the
same library lock before reading or changing either side of the projection.
"""
from psycopg.rows import tuple_row


def lock_library(connection, library_id):
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                       (f"source-visibility:{library_id}",))


def rebuild_links(connection, library_id, user_ids):
    """Rebuild only these owners, from proven memberships, under library lock."""
    with connection.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(
            """DELETE FROM user_track_links
               WHERE library_id=%s AND user_id=ANY(%s::uuid[])
               RETURNING user_id, track_id""", (library_id, user_ids))
        before = set(cursor.fetchall())
        cursor.execute(
            """INSERT INTO user_track_links (user_id,library_id,track_id,external_id)
               SELECT DISTINCT ON (m.user_id,ts.track_id)
                   m.user_id,m.library_id,ts.track_id,m.external_id
               FROM user_source_memberships m
               JOIN track_sources ts ON ts.library_id=m.library_id
                 AND ts.source_type='subsonic' AND ts.external_id=m.external_id
               WHERE m.library_id=%s AND m.user_id=ANY(%s::uuid[])
               ORDER BY m.user_id,ts.track_id,m.external_id
               RETURNING user_id,track_id""", (library_id, user_ids))
        after = set(cursor.fetchall())
    return {"linked": len(after), "unlinked": len(before - after)}


def update_memberships(connection, library_id, user_id, external_ids, *, full=False):
    """Full catalogs replace membership; batches add without removing siblings.

    IDs need not have a track_sources row yet: identity discovery projects them
    later, without granting access to anyone else.
    """
    lock_library(connection, library_id)
    with connection.cursor() as cursor:
        if full:
            cursor.execute(
                """DELETE FROM user_source_memberships
                   WHERE library_id=%s AND user_id=%s
                     AND NOT (external_id=ANY(%s::text[]))""",
                (library_id, user_id, external_ids))
        cursor.execute(
            """INSERT INTO user_source_memberships(user_id,library_id,external_id)
               SELECT %s,%s,unnest(%s::text[]) ON CONFLICT DO NOTHING""",
            (user_id, library_id, external_ids))
    return rebuild_links(connection, library_id, [user_id])


def source_remapped(connection, library_id, external_id):
    """Caller must hold library lock from before its track_sources write."""
    with connection.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(
            """SELECT user_id FROM user_source_memberships
               WHERE library_id=%s AND external_id=%s""", (library_id, external_id))
        owners = [row[0] for row in cursor.fetchall()]
    if owners:
        rebuild_links(connection, library_id, owners)
