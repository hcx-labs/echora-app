"""Replace password authentication with OIDC identities.

Revision ID: 0007_oidc_auth
"""
from alembic import op

revision = "0007_oidc_auth"
down_revision = "0006_time_curations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email text")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS oidc_subject text")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_blocked boolean NOT NULL DEFAULT false")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx ON users (lower(email)) WHERE email IS NOT NULL")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_oidc_subject_idx ON users (oidc_subject) WHERE oidc_subject IS NOT NULL")
    op.execute("""CREATE TABLE IF NOT EXISTS oidc_settings (
        singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
        auto_provision boolean NOT NULL DEFAULT true,
        updated_at timestamptz NOT NULL DEFAULT now()
    )""")
    op.execute("INSERT INTO oidc_settings (singleton) VALUES (true) ON CONFLICT DO NOTHING")
    op.execute("""CREATE TABLE IF NOT EXISTS oidc_allowed_emails (
        email text PRIMARY KEY,
        created_by uuid REFERENCES users(id) ON DELETE SET NULL,
        created_at timestamptz NOT NULL DEFAULT now()
    )""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS oidc_allowed_emails")
    op.execute("DROP TABLE IF EXISTS oidc_settings")
    op.execute("DROP INDEX IF EXISTS users_oidc_subject_idx")
    op.execute("DROP INDEX IF EXISTS users_email_idx")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS is_blocked")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS is_admin")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS oidc_subject")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS email")
