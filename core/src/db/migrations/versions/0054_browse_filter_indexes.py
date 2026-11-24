"""Indexes for scoped browse filtering. Existing title/year/visibility indexes are reused."""
from alembic import op

revision = '0054_browse_filter_indexes'
down_revision = '0053_singable_translation_prompt'
branch_labels = None
depends_on = None

INDEXES = {
    'tracks_browse_genres_idx': 'ON tracks USING gin (genres)',
    'lyrics_browse_language_track_idx': 'ON lyrics (language, track_id)',
    'lyrics_browse_available_idx': "ON lyrics (track_id) WHERE NULLIF(btrim(text),'') IS NOT NULL",
    'lyrics_browse_ai_idx': "ON lyrics (track_id) WHERE provenance->>'ai_generated'='true'",
    'translations_browse_ready_idx': "ON lyric_translations (target_language, track_id) WHERE status='ready'",
    'embeddings_browse_voice_idx': "ON embeddings (track_id, run_id) WHERE embedding_type='voice-gender' AND window_index IS NULL",
}


def upgrade():
    with op.get_context().autocommit_block():
        for name, definition in INDEXES.items():
            op.execute(f'CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} {definition}')


def downgrade():
    with op.get_context().autocommit_block():
        for name in INDEXES:
            op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS {name}')
