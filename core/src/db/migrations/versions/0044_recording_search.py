"""Versioned recording fingerprints and private, expiring recording queries."""
from alembic import op

revision = "0044_recording_search"
down_revision = "0043_visual_feature_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
CREATE TABLE recording_fingerprints (
    track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    representation_id text NOT NULL,
    segment_count integer NOT NULL CHECK (segment_count >= 0),
    fingerprints bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (track_id, representation_id),
    CHECK (octet_length(fingerprints) = segment_count * 128 * 4)
);
CREATE INDEX recording_fingerprints_representation ON recording_fingerprints(representation_id, track_id);
CREATE TABLE recording_searches (
    job_id uuid PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    representation_id text NOT NULL,
    policy_id text NOT NULL,
    audio bytea CHECK (octet_length(audio) <= 8388608),
    result jsonb,
    audio_expires_at timestamptz NOT NULL DEFAULT now() + interval '15 minutes',
    expires_at timestamptz NOT NULL DEFAULT now() + interval '1 day'
);
CREATE INDEX recording_searches_expiry ON recording_searches(expires_at);
CREATE FUNCTION erase_recording_query_audio() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.kind = 'recording_search' AND NEW.status IN ('complete','partial','failed','cancelled') THEN
        UPDATE recording_searches SET audio=NULL WHERE job_id=NEW.id;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER recording_query_terminal AFTER UPDATE OF status ON jobs
    FOR EACH ROW EXECUTE FUNCTION erase_recording_query_audio();
""")


def downgrade() -> None:
    op.execute("DROP TRIGGER recording_query_terminal ON jobs")
    op.execute("DROP FUNCTION erase_recording_query_audio()")
    op.drop_table("recording_searches")
    op.drop_table("recording_fingerprints")
