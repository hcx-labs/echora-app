"""Track rolling idle expiry separately from a fixed login lifetime."""
from alembic import op

revision = "0050_session_activity"
down_revision = "0049_job_dismissal"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE user_sessions ADD COLUMN absolute_expires_at timestamptz")
    op.execute("ALTER TABLE user_sessions ADD COLUMN last_renewed_at timestamptz")
    # Do not silently extend or revive legacy sessions. The next OIDC login
    # creates a session with the configured idle and absolute limits.
    op.execute("UPDATE user_sessions SET absolute_expires_at=expires_at, last_renewed_at=created_at")
    op.execute("ALTER TABLE user_sessions ALTER COLUMN absolute_expires_at SET NOT NULL")
    op.execute("ALTER TABLE user_sessions ALTER COLUMN last_renewed_at SET NOT NULL")
    op.execute("ALTER TABLE user_sessions ADD CONSTRAINT session_idle_within_absolute CHECK (expires_at <= absolute_expires_at)")


def downgrade():
    op.execute("ALTER TABLE user_sessions DROP CONSTRAINT session_idle_within_absolute")
    op.execute("ALTER TABLE user_sessions DROP COLUMN last_renewed_at")
    op.execute("ALTER TABLE user_sessions DROP COLUMN absolute_expires_at")
