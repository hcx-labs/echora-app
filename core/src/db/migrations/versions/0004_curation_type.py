"""Distinguish curation recipe types.

Revision ID: 0004_curation_type
"""
from alembic import op

revision = "0004_curation_type"
down_revision = "0003_curation_recipe"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS curation_type text NOT NULL DEFAULT 'language'")
    op.execute("ALTER TABLE curations ADD CONSTRAINT curations_type_check CHECK (curation_type IN ('language', 'examples', 'time_of_day'))")


def downgrade() -> None:
    op.execute("ALTER TABLE curations DROP CONSTRAINT IF EXISTS curations_type_check")
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS curation_type")
