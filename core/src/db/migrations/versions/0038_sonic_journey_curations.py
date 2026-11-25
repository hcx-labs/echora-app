"""Sonic journey curations: ordered waypoints through embedding space."""
from alembic import op

revision = "0038_sonic_journey_curations"
down_revision = "0037_semantic_fusion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE curations DROP CONSTRAINT curations_type_check;\n"
        "ALTER TABLE curations ADD CONSTRAINT curations_type_check\n"
        "  CHECK (curation_type = ANY (ARRAY['combined'::text, 'language'::text, 'examples'::text,\n"
        "    'time_of_day'::text, 'sonic_journey'::text]));\n"
        "ALTER TABLE curations\n"
        "  ADD COLUMN IF NOT EXISTS journey_start_track_id uuid REFERENCES tracks(id) ON DELETE SET NULL,\n"
        "  ADD COLUMN IF NOT EXISTS journey_stop_track_ids uuid[] NOT NULL DEFAULT '{}',\n"
        "  ADD COLUMN IF NOT EXISTS journey_end_track_id uuid REFERENCES tracks(id) ON DELETE SET NULL,\n"
        "  ADD COLUMN IF NOT EXISTS journey_lyrics_weight smallint NOT NULL DEFAULT 25;"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE curations DROP CONSTRAINT curations_type_check;\n"
        "ALTER TABLE curations ADD CONSTRAINT curations_type_check\n"
        "  CHECK (curation_type = ANY (ARRAY['combined'::text, 'language'::text, 'examples'::text,\n"
        "    'time_of_day'::text]));\n"
        "ALTER TABLE curations\n"
        "  DROP COLUMN IF EXISTS journey_start_track_id,\n"
        "  DROP COLUMN IF EXISTS journey_stop_track_ids,\n"
        "  DROP COLUMN IF EXISTS journey_end_track_id,\n"
        "  DROP COLUMN IF EXISTS journey_lyrics_weight;"
    )
