"""Add structured curation prompt channels.

Revision ID: 0024_structured_curation_prompts
Revises: 0023_hum_processing_settings
"""
from alembic import op

revision = "0024_structured_curation_prompts"
down_revision = "0023_hum_processing_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE curations
      ADD COLUMN IF NOT EXISTS sound_prompt text NOT NULL DEFAULT '',
      ADD COLUMN IF NOT EXISTS themes_prompt text NOT NULL DEFAULT '',
      ADD COLUMN IF NOT EXISTS sound_weight smallint NOT NULL DEFAULT 50""")


def downgrade() -> None:
    op.execute("""ALTER TABLE curations
      DROP COLUMN IF EXISTS sound_prompt,
      DROP COLUMN IF EXISTS themes_prompt,
      DROP COLUMN IF EXISTS sound_weight""")
