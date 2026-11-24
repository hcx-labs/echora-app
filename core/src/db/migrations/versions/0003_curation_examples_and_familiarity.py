"""Add explicit curation examples and the future familiarity mix.

Revision ID: 0003_curation_recipe
"""
from alembic import op

revision = "0003_curation_recipe"
down_revision = "0002_query_audit_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS positive_track_ids uuid[] NOT NULL DEFAULT '{}'")
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS negative_track_ids uuid[] NOT NULL DEFAULT '{}'")
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS familiarity_percent integer NOT NULL DEFAULT 70")
    op.execute("ALTER TABLE curations ADD CONSTRAINT curations_familiarity_percent_check CHECK (familiarity_percent BETWEEN 0 AND 100)")


def downgrade() -> None:
    op.execute("ALTER TABLE curations DROP CONSTRAINT IF EXISTS curations_familiarity_percent_check")
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS familiarity_percent")
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS negative_track_ids")
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS positive_track_ids")
