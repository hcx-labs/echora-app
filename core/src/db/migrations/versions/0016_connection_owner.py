"""Scope Navidrome connections to their owning user.

Revision ID: 0016_connection_owner
Revises: 0015_animation_speed
"""
from alembic import op

revision = "0016_connection_owner"
down_revision = "0015_animation_speed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE navidrome_connections "
        "ADD COLUMN IF NOT EXISTS owner_user_id uuid REFERENCES users(id) ON DELETE CASCADE"
    )
    op.execute("CREATE INDEX IF NOT EXISTS navidrome_connections_owner_idx ON navidrome_connections (owner_user_id)")
    op.execute(
        "UPDATE navidrome_connections c "
        "SET owner_user_id = backfill.owner_id "
        "FROM ("
        "  SELECT c.id AS connection_id, min(p.user_id::text)::uuid AS owner_id "
        "  FROM navidrome_connections c "
        "  JOIN user_preferences p ON p.navidrome_connection_id = c.id "
        "  GROUP BY c.id"
        ") AS backfill "
        "WHERE c.id = backfill.connection_id AND c.owner_user_id IS NULL"
    )
    op.execute("ALTER TABLE navidrome_connections DROP CONSTRAINT IF EXISTS navidrome_connections_url_username_key")
    op.execute(
        "ALTER TABLE navidrome_connections "
        "ADD CONSTRAINT navidrome_connections_owner_url_username_key "
        "UNIQUE (owner_user_id, url, username)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE navidrome_connections DROP CONSTRAINT IF EXISTS navidrome_connections_owner_url_username_key")
    op.execute(
        "ALTER TABLE navidrome_connections "
        "ADD CONSTRAINT navidrome_connections_url_username_key UNIQUE (url, username)"
    )
    op.execute("DROP INDEX IF EXISTS navidrome_connections_owner_idx")
    op.execute("ALTER TABLE navidrome_connections DROP COLUMN IF EXISTS owner_user_id")
