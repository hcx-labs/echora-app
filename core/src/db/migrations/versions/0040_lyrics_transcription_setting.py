"""Opt-in AI lyrics transcription."""
from alembic import op

revision = '0040_lyrics_transcription'
down_revision = '0039_curation_sound_profiles'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('ALTER TABLE analysis_settings ADD COLUMN IF NOT EXISTS transcription_processing_enabled boolean NOT NULL DEFAULT false')


def downgrade():
    op.execute('ALTER TABLE analysis_settings DROP COLUMN IF EXISTS transcription_processing_enabled')
