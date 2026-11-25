"""Owner-scoped, consented recording calibration samples stored on private disk."""
from alembic import op

revision = "0045_recording_calibration"
down_revision = "0044_recording_search"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
CREATE TABLE recording_calibration_samples (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expected_track_id uuid,
    not_in_library boolean NOT NULL DEFAULT false,
    notes text NOT NULL DEFAULT '' CHECK (char_length(notes) <= 300),
    consent_version text NOT NULL CHECK (consent_version='save-for-calibration-v1'),
    audio_sha256 text NOT NULL CHECK (char_length(audio_sha256)=64),
    byte_count integer NOT NULL CHECK (byte_count BETWEEN 1 AND 8388608),
    duration_seconds double precision NOT NULL CHECK (duration_seconds BETWEEN 1 AND 20),
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL DEFAULT now() + interval '7 days',
    CHECK ((expected_track_id IS NOT NULL) <> not_in_library)
);
CREATE INDEX recording_calibration_owner ON recording_calibration_samples(user_id,created_at DESC);
CREATE INDEX recording_calibration_expiry ON recording_calibration_samples(expires_at);
""")


def downgrade() -> None:
    # Operators should run the calibration cleanup before downgrading if retained
    # raw recordings must be erased immediately. Files are not database objects.
    op.drop_table("recording_calibration_samples")
