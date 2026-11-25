"""Explicitly armed, one-shot private recording diagnostics with 24-hour expiry."""
from alembic import op

revision = "0046_recording_diagnostics"
down_revision = "0045_recording_calibration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
CREATE TABLE recording_diagnostic_captures (
    user_id uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    consent_version text NOT NULL CHECK (consent_version='preserve-next-recording-v1'),
    armed_at timestamptz NOT NULL DEFAULT now(),
    captured_at timestamptz,
    expires_at timestamptz NOT NULL DEFAULT now()+interval '24 hours',
    job_id uuid REFERENCES jobs(id) ON DELETE SET NULL,
    audio bytea,
    CHECK (audio IS NULL OR octet_length(audio) BETWEEN 1 AND 8388608),
    CHECK ((captured_at IS NULL AND audio IS NULL AND job_id IS NULL)
        OR (captured_at IS NOT NULL AND audio IS NOT NULL))
);
CREATE INDEX recording_diagnostic_expiry ON recording_diagnostic_captures(expires_at);
""")


def downgrade() -> None:
    op.drop_table("recording_diagnostic_captures")
