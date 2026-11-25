"""Move animation speed to client-side storage.

Revision ID: 0017_drop_animation_speed
Revises: 0016_connection_owner
"""
from alembic import op

revision = "0017_drop_animation_speed"
down_revision = "0016_connection_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE user_preferences DROP COLUMN IF EXISTS animation_speed")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE user_preferences "
        "ADD COLUMN IF NOT EXISTS animation_speed text NOT NULL DEFAULT 'normal' "
        "CHECK (animation_speed IN ('slow', 'normal', 'fast'))"
    )
