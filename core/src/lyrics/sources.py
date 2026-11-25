"""Durable sync pause and original provider snapshots, separate from enriched lyrics."""

from psycopg.types.json import Jsonb


def lyrics_paused(db, url):
    # Scope by server URL, not key owner: another account syncing the same server
    # must also bypass the instance-wide plugin. Parents cover gaps between batches.
    row = db.execute(
        """SELECT EXISTS (
            SELECT 1 FROM jobs j
            JOIN navidrome_connections n ON n.id=j.connection_id
            WHERE rtrim(n.url,'/')=%s AND j.status IN ('queued','running','waiting')
                AND (j.kind IN ('navidrome_sync','import','lyrics_backfill','karaoke_backfill')
                     OR (j.kind='analysis_batch' AND j.payload->>'operation'
                         IN ('navidrome_sync','import','lyrics_backfill','karaoke_backfill')))
        ) AS paused""",
        (url.rstrip("/"),),
    ).fetchone()
    return row["paused"]


def store_original(db, library_id, external_id, track_id, result):
    # Keep the last successful original even when a later fetch is missing or
    # unavailable. Raw structured tracks include ends, cues and language variants.
    available = result if str(result.get("text") or "").strip() else None
    db.execute(
        """INSERT INTO navidrome_lyrics_sources
            (library_id,external_id,track_id,result,last_available)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (library_id,external_id) DO UPDATE SET track_id=EXCLUDED.track_id,
                result=EXCLUDED.result,
                last_available=coalesce(EXCLUDED.last_available,navidrome_lyrics_sources.last_available),
                fetched_at=now()""",
        (library_id, external_id, track_id, Jsonb(result), Jsonb(available) if available else None),
    )
