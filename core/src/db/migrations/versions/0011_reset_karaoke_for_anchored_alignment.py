"""Reset karaoke results for source-timeline-anchored alignment.

Revision ID: 0011_anchored_karaoke
Revises: 0010_karaoke_variants
"""
from alembic import op

revision = "0011_anchored_karaoke"
down_revision = "0010_karaoke_variants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM karaoke_lyrics_variants")
    op.execute("""UPDATE lyrics
      SET karaoke_lines=NULL, karaoke_ass=NULL, karaoke_lrc=NULL,
          karaoke_model=NULL, karaoke_model_revision=NULL, karaoke_bounded=NULL,
          karaoke_created_at=NULL, provenance=provenance - 'karaoke'
      WHERE karaoke_lines IS NOT NULL OR provenance ? 'karaoke'""")


def downgrade() -> None:
    pass
