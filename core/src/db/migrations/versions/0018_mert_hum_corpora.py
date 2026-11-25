"""Add experimental MERT humming corpora.

Revision ID: 0018_mert_hum_corpora
Revises: 0017_drop_animation_speed
"""
from alembic import op

revision = "0018_mert_hum_corpora"
down_revision = "0017_drop_animation_speed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE hum_corpora (
          id uuid PRIMARY KEY,
          user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          run_id uuid REFERENCES analysis_runs(id) ON DELETE CASCADE,
          status text NOT NULL CHECK (status IN ('building', 'complete', 'failed')),
          track_limit integer NOT NULL CHECK (track_limit BETWEEN 1 AND 500),
          error text,
          created_at timestamptz NOT NULL DEFAULT now(),
          completed_at timestamptz
        )
    """)
    op.execute("CREATE INDEX hum_corpora_user_created_idx ON hum_corpora (user_id, created_at DESC)")
    op.execute("""
        CREATE TABLE hum_corpus_tracks (
          corpus_id uuid NOT NULL REFERENCES hum_corpora(id) ON DELETE CASCADE,
          track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
          PRIMARY KEY (corpus_id, track_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS hum_corpus_tracks")
    op.execute("DROP TABLE IF EXISTS hum_corpora")
