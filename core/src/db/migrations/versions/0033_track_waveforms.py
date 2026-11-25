"""Store precomputed whole-track waveform peaks."""
from alembic import op

revision = "0033_track_waveforms"
down_revision = "0032_audio_descriptors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE IF NOT EXISTS track_waveforms (
  track_id uuid PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
  revision text NOT NULL,
  duration_seconds double precision NOT NULL CHECK (duration_seconds > 0),
  peaks jsonb NOT NULL CHECK (jsonb_typeof(peaks) = 'array'),
  created_at timestamptz NOT NULL DEFAULT now()
);
""")


def downgrade() -> None:
    op.drop_table("track_waveforms")
