"""Per-song resync requests, and whether a full sync realigns karaoke of an older revision."""

from alembic import op

revision = "0067_track_resync_requests"
down_revision = "0066_karaoke_stems"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        -- Off: a full sync aligns songs without karaoke but keeps alignments of older pipeline
        -- or model revisions; songs are realigned when asked (track_resync_requests).
        ALTER TABLE analysis_settings
            ADD COLUMN karaoke_realign_outdated boolean NOT NULL DEFAULT true;
        -- A song's pipeline the next sync (partial or full) redoes, whatever its revision.
        CREATE TABLE track_resync_requests (
            track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            pipeline text NOT NULL CHECK (
                pipeline IN ('transcription', 'karaoke', 'karaoke_stems', 'translation', 'themes')
            ),
            options jsonb NOT NULL DEFAULT '{}'::jsonb,
            requested_by uuid REFERENCES users(id) ON DELETE SET NULL,
            requested_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (track_id, pipeline)
        );
    """)


def downgrade():
    op.execute("""DROP TABLE track_resync_requests;
        ALTER TABLE analysis_settings DROP COLUMN karaoke_realign_outdated""")
