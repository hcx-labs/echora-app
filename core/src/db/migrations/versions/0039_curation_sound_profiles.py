"""Persist curation sound-profile targets."""
from alembic import op

revision = "0039_curation_sound_profiles"
down_revision = "0038_sonic_journey_curations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS sound_profile jsonb NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS sound_profile")
