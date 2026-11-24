"""Relax refresh interval constraint and default to 24 hours.

Revision ID: 0028_refresh_interval_24h
Revises: 0027_curation_language_target
"""
from alembic import op

revision = "0028_refresh_interval_24h"
down_revision = "0027_curation_language_target"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE curations DROP CONSTRAINT IF EXISTS curations_refresh_interval_hours_check")
    op.execute("ALTER TABLE curations ALTER COLUMN refresh_interval_hours SET DEFAULT 24")
    op.execute("UPDATE curations SET refresh_interval_hours = 24 WHERE refresh_interval_hours = 6")


def downgrade() -> None:
    op.execute("UPDATE curations SET refresh_interval_hours = 6 WHERE refresh_interval_hours = 24")
    op.execute("ALTER TABLE curations ALTER COLUMN refresh_interval_hours SET DEFAULT 6")
    op.execute("ALTER TABLE curations ADD CONSTRAINT curations_refresh_interval_hours_check CHECK (refresh_interval_hours = 6)")
