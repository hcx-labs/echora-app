"""Archive Navidrome source lyrics independently from enriched lyrics."""

from alembic import op

revision = "0056_navidrome_original_lyrics"
down_revision = "0055_navidrome_integration"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE navidrome_lyrics_sources (
            library_id uuid NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
            external_id text NOT NULL,
            track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            result jsonb NOT NULL,
            last_available jsonb,
            fetched_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (library_id, external_id)
        );
        CREATE INDEX jobs_active_lyrics_connection ON jobs(connection_id)
            WHERE status IN ('queued','running','waiting')
            AND (kind IN ('navidrome_sync','import','lyrics_backfill','karaoke_backfill')
                 OR (kind='analysis_batch' AND payload->>'operation'
                     IN ('navidrome_sync','import','lyrics_backfill','karaoke_backfill')));
    """)


def downgrade():
    op.execute("DROP INDEX jobs_active_lyrics_connection; DROP TABLE navidrome_lyrics_sources")
