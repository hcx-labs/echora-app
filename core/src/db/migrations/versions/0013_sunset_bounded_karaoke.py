"""Sunset bounded karaoke and add a processing toggle.

Revision ID: 0013_karaoke_toggle
Revises: 0012_karaoke_documents
"""
from alembic import op

revision = "0013_karaoke_toggle"
down_revision = "0012_karaoke_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE analysis_settings
      ADD COLUMN IF NOT EXISTS karaoke_processing_enabled boolean NOT NULL DEFAULT true""")
    op.execute("UPDATE analysis_settings SET karaoke_bound_to_synced_lines=false")
    op.execute("DELETE FROM karaoke_lyrics_variants WHERE bounded=true")


def downgrade() -> None:
    op.execute("ALTER TABLE analysis_settings DROP COLUMN IF EXISTS karaoke_processing_enabled")
