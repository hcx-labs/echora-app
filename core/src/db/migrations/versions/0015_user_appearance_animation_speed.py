"""Store a per-user application animation speed.

Revision ID: 0015_animation_speed
Revises: 0014_karaoke_v1
"""
from alembic import op

revision = "0015_animation_speed"
down_revision = "0014_karaoke_v1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE user_preferences "
        "ADD COLUMN IF NOT EXISTS animation_speed text NOT NULL DEFAULT 'normal' "
        "CHECK (animation_speed IN ('slow', 'normal', 'fast'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE user_preferences DROP COLUMN IF EXISTS animation_speed")
