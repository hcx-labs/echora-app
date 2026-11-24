"""Persist proven user source access independently of canonical track links.

Previously discarded aliases cannot be recovered here: a normal full authorized
catalog scan is required. Never infer ownership from global track_sources.
"""
from alembic import op

revision = "0047_source_membership"
down_revision = "0046_recording_diagnostics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
CREATE TABLE user_source_memberships (
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    library_id uuid NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    external_id text NOT NULL,
    PRIMARY KEY (user_id, library_id, external_id)
);
CREATE INDEX user_source_memberships_source
    ON user_source_memberships(library_id, external_id);
INSERT INTO user_source_memberships(user_id,library_id,external_id)
SELECT user_id,library_id,external_id FROM user_track_links
WHERE external_id IS NOT NULL
ON CONFLICT DO NOTHING;
""")


def downgrade() -> None:
    op.drop_table("user_source_memberships")
