"""User-scoped Navidrome ranking profiles and revocable machine credentials."""

from alembic import op

revision = "0055_navidrome_integration"
down_revision = "0054_browse_filter_indexes"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE navidrome_integrations (
            user_id uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            enabled boolean NOT NULL DEFAULT false,
            connection_id uuid REFERENCES navidrome_connections(id) ON DELETE CASCADE,
            profile jsonb NOT NULL DEFAULT '{}',
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE navidrome_api_keys (
            id uuid PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES navidrome_integrations(user_id) ON DELETE CASCADE,
            connection_id uuid NOT NULL REFERENCES navidrome_connections(id) ON DELETE CASCADE,
            label text NOT NULL,
            prefix text NOT NULL,
            token_hash text NOT NULL UNIQUE,
            created_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL,
            revoked_at timestamptz,
            last_used_at timestamptz,
            CHECK (expires_at > created_at)
        );
        CREATE INDEX navidrome_api_keys_owner ON navidrome_api_keys(user_id, created_at DESC);
    """)


def downgrade():
    op.execute("DROP TABLE navidrome_api_keys; DROP TABLE navidrome_integrations")
