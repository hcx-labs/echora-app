"""Replace singular curation prompts with per-channel tag lists.

Revision ID: 0025_curation_prompt_tags
Revises: 0024_structured_curation_prompts
"""
from alembic import op

revision = "0025_curation_prompt_tags"
down_revision = "0024_structured_curation_prompts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE curations
      DROP COLUMN IF EXISTS sound_prompt,
      DROP COLUMN IF EXISTS themes_prompt,
      DROP COLUMN IF EXISTS sound_weight""")
    op.execute("""ALTER TABLE curations
      ADD COLUMN IF NOT EXISTS sound_prompts text[] NOT NULL DEFAULT '{}',
      ADD COLUMN IF NOT EXISTS themes_prompts text[] NOT NULL DEFAULT '{}',
      ADD COLUMN IF NOT EXISTS sound_negative_prompts text[] NOT NULL DEFAULT '{}',
      ADD COLUMN IF NOT EXISTS themes_negative_prompts text[] NOT NULL DEFAULT '{}',
      ADD COLUMN IF NOT EXISTS sound_weight smallint NOT NULL DEFAULT 50""")


def downgrade() -> None:
    op.execute("""ALTER TABLE curations
      DROP COLUMN IF EXISTS sound_prompts,
      DROP COLUMN IF EXISTS themes_prompts,
      DROP COLUMN IF EXISTS sound_negative_prompts,
      DROP COLUMN IF EXISTS themes_negative_prompts,
      DROP COLUMN IF EXISTS sound_weight""")
    op.execute("""ALTER TABLE curations
      ADD COLUMN IF NOT EXISTS sound_prompt text NOT NULL DEFAULT '',
      ADD COLUMN IF NOT EXISTS themes_prompt text NOT NULL DEFAULT '',
      ADD COLUMN IF NOT EXISTS sound_weight smallint NOT NULL DEFAULT 50""")
