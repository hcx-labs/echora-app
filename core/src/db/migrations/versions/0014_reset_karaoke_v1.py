"""Promote the validated karaoke pipeline to the stable v1 baseline.

Revision ID: 0014_karaoke_v1
Revises: 0013_karaoke_toggle
"""
from alembic import op

revision = "0014_karaoke_v1"
down_revision = "0013_karaoke_toggle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM karaoke_lyrics_variants")
    op.execute("""UPDATE lyrics
      SET karaoke_lines=NULL, karaoke_ass=NULL, karaoke_lrc=NULL,
          karaoke_model=NULL, karaoke_model_revision=NULL, karaoke_bounded=NULL,
          karaoke_created_at=NULL, provenance=provenance-'karaoke'
      WHERE karaoke_lines IS NOT NULL OR provenance ? 'karaoke'""")


def downgrade() -> None:
    pass
