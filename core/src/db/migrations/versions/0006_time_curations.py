"""Store recurring time-of-day recipe fields.

Revision ID: 0006_time_curations
"""
from alembic import op

revision = "0006_time_curations"
down_revision = "0005_user_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS period_start text")
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS period_end text")
    op.execute("ALTER TABLE curations ADD COLUMN IF NOT EXISTS lookback_days integer NOT NULL DEFAULT 7")
    op.execute("ALTER TABLE curations ADD CONSTRAINT curations_lookback_days_check CHECK (lookback_days BETWEEN 1 AND 90)")


def downgrade() -> None:
    op.execute("ALTER TABLE curations DROP CONSTRAINT IF EXISTS curations_lookback_days_check")
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS lookback_days")
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS period_end")
    op.execute("ALTER TABLE curations DROP COLUMN IF EXISTS period_start")
