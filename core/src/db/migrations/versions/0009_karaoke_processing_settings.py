"""Add global karaoke processing settings.

Revision ID: 0009_karaoke_settings
Revises: 0008_karaoke_lyrics
"""
from alembic import op

revision = "0009_karaoke_settings"
down_revision = "0008_karaoke_lyrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE IF NOT EXISTS analysis_settings (
      singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
      karaoke_bound_to_synced_lines boolean NOT NULL DEFAULT true,
      updated_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("INSERT INTO analysis_settings (singleton) VALUES (true) ON CONFLICT DO NOTHING")
    op.execute("ALTER TABLE lyrics ADD COLUMN IF NOT EXISTS karaoke_bounded boolean")
    op.execute("UPDATE lyrics SET karaoke_bounded=false WHERE karaoke_model='fa_kara' AND karaoke_bounded IS NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE lyrics DROP COLUMN IF EXISTS karaoke_bounded")
    op.execute("DROP TABLE IF EXISTS analysis_settings")
