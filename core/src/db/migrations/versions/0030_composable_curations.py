"""Allow curation aspects to be combined.

Revision ID: 0030_composable_curations
Revises: 0029_audio_profiles
"""
from alembic import op

revision = "0030_composable_curations"
down_revision = "0029_audio_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE curations DROP CONSTRAINT IF EXISTS curations_type_check")
    op.execute("""ALTER TABLE curations ADD CONSTRAINT curations_type_check
      CHECK (curation_type IN ('combined', 'language', 'examples', 'time_of_day'))""")
    op.execute("""ALTER TABLE curations ADD COLUMN IF NOT EXISTS
      time_of_day_enabled boolean NOT NULL DEFAULT false""")


def downgrade() -> None:
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS time_of_day_enabled")
    op.execute("ALTER TABLE curations DROP CONSTRAINT IF EXISTS curations_type_check")
    op.execute("UPDATE curations SET curation_type='language' WHERE curation_type='combined'")
    op.execute("""ALTER TABLE curations ADD CONSTRAINT curations_type_check
      CHECK (curation_type IN ('language', 'examples', 'time_of_day'))""")
