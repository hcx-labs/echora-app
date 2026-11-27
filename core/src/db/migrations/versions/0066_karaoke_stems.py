"""Karaoke stems: each song's vocals and instrumental, encoded for the Stage player to remix."""

from alembic import op

revision = "0066_karaoke_stems"
down_revision = "0065_transcription_model"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE analysis_settings
            ADD COLUMN karaoke_stems_processing_enabled boolean NOT NULL DEFAULT false,
            ADD COLUMN karaoke_stems_modal_enabled boolean NOT NULL DEFAULT false;
        -- One stems file per song. A new file gets a new id, so a URL's content never changes.
        CREATE TABLE karaoke_stems (
            id uuid PRIMARY KEY,
            track_id uuid NOT NULL UNIQUE REFERENCES tracks(id) ON DELETE CASCADE,
            recipe text NOT NULL,
            status text NOT NULL CHECK (status IN ('complete', 'failed')),
            path text,
            bytes bigint,
            duration_seconds real,
            error text,
            created_at timestamptz NOT NULL DEFAULT now(),
            CHECK (status <> 'complete' OR path IS NOT NULL)
        );
    """)


def downgrade():
    # Encoded files under ECHORA_KARAOKE_STEMS_DIR are left on disk.
    op.execute("""DROP TABLE karaoke_stems;
        ALTER TABLE analysis_settings DROP COLUMN karaoke_stems_processing_enabled,
            DROP COLUMN karaoke_stems_modal_enabled""")
