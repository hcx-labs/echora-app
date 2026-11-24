"""Add the application-wide hum processing setting.

Revision ID: 0023_hum_processing_settings
Revises: 0022_drop_melody_windows
"""
from alembic import op

revision = "0023_hum_processing_settings"
down_revision = "0022_drop_melody_windows"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE analysis_settings
      ADD COLUMN IF NOT EXISTS hum_processing_enabled boolean NOT NULL DEFAULT true""")


def downgrade() -> None:
    op.execute("ALTER TABLE analysis_settings DROP COLUMN IF EXISTS hum_processing_enabled")
