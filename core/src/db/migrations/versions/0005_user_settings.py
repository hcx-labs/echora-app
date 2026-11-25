"""Add user timezone and Last.fm connection settings.

Revision ID: 0005_user_settings
"""
from alembic import op

revision = "0005_user_settings"
down_revision = "0004_curation_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS timezone text NOT NULL DEFAULT 'UTC'")
    op.execute("ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS lastfm_username text")
    op.execute("ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS lastfm_api_key_encrypted bytea")


def downgrade() -> None:
    op.execute("ALTER TABLE user_preferences DROP COLUMN IF EXISTS lastfm_api_key_encrypted")
    op.execute("ALTER TABLE user_preferences DROP COLUMN IF EXISTS lastfm_username")
    op.execute("ALTER TABLE user_preferences DROP COLUMN IF EXISTS timezone")
