"""Add an ANN index for melody contour windows.

Revision ID: 0021_melody_window_index
Revises: 0020_multiple_melody_sources
"""
from alembic import op

revision = "0021_melody_window_index"
down_revision = "0020_multiple_melody_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE melody_windows (
          id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
          run_id uuid NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
          source text NOT NULL,
          start_seconds real NOT NULL CHECK (start_seconds >= 0),
          duration_seconds real NOT NULL CHECK (duration_seconds > 0),
          embedding vector(192) NOT NULL,
          UNIQUE (track_id, run_id, source, start_seconds, duration_seconds)
        )
    """)
    op.execute("CREATE INDEX melody_windows_run_idx ON melody_windows (run_id)")
    op.execute("CREATE INDEX melody_windows_embedding_hnsw_idx ON melody_windows USING hnsw (embedding vector_cosine_ops)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS melody_windows")
