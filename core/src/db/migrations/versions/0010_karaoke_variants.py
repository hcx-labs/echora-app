"""Retain bounded and unbounded karaoke variants.

Revision ID: 0010_karaoke_variants
Revises: 0009_karaoke_settings
"""
from alembic import op

revision = "0010_karaoke_variants"
down_revision = "0009_karaoke_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE IF NOT EXISTS karaoke_lyrics_variants (
      track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
      bounded boolean NOT NULL,
      lines jsonb NOT NULL,
      ass text,
      lrc text,
      model text NOT NULL,
      model_revision text NOT NULL,
      provenance jsonb NOT NULL DEFAULT '{}',
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY (track_id, bounded)
    )""")
    op.execute("""INSERT INTO karaoke_lyrics_variants
      (track_id, bounded, lines, ass, lrc, model, model_revision, provenance, created_at)
    SELECT track_id, coalesce(karaoke_bounded, false), karaoke_lines, karaoke_ass, karaoke_lrc,
           coalesce(karaoke_model, 'fa_kara'), coalesce(karaoke_model_revision, 'unknown'),
           coalesce(provenance->'karaoke', '{}'::jsonb), coalesce(karaoke_created_at, now())
    FROM lyrics WHERE karaoke_lines IS NOT NULL
    ON CONFLICT (track_id, bounded) DO NOTHING""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS karaoke_lyrics_variants")
