"""Persist revisioned visualization timelines on existing installations."""
from alembic import op

revision = "0042_track_visual_features"
down_revision = "0041_manual_lyrics_overrides"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE IF NOT EXISTS track_visual_features (
        track_id uuid PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
        revision text NOT NULL,
        status text NOT NULL DEFAULT 'complete' CHECK (status IN ('complete', 'unsupported')),
        duration_seconds double precision NOT NULL CHECK (duration_seconds > 0),
        hop_seconds real NOT NULL CHECK (hop_seconds > 0),
        features jsonb NOT NULL CHECK (jsonb_typeof(features) = 'object'),
        created_at timestamptz NOT NULL DEFAULT now()
    )""")


def downgrade():
    op.drop_table("track_visual_features")
