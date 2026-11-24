"""Add melody contours for query-by-humming.

Revision ID: 0019_melody_contours
Revises: 0018_mert_hum_corpora
"""
from alembic import op

revision = "0019_melody_contours"
down_revision = "0018_mert_hum_corpora"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE melody_contours (
          id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
          run_id uuid NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
          source text NOT NULL CHECK (source IN ('predominant-melody', 'hum-query')),
          hop_seconds real NOT NULL CHECK (hop_seconds > 0),
          pitch real[] NOT NULL,
          voiced boolean[] NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          CHECK (cardinality(pitch) = cardinality(voiced)),
          UNIQUE (track_id, run_id, source)
        )
    """)
    op.execute("CREATE INDEX melody_contours_run_track_idx ON melody_contours (run_id, track_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS melody_contours")
