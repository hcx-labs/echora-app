"""Add indexes found by the first query-plan audit.

Revision ID: 0002_query_audit_indexes
"""
from alembic import op

revision = "0002_query_audit_indexes"
down_revision = "0001_existing_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE INDEX IF NOT EXISTS tracks_title_trgm_idx ON tracks USING gin (title gin_trgm_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS tracks_artist_trgm_idx ON tracks USING gin (artist gin_trgm_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS tracks_album_trgm_idx ON tracks USING gin (album gin_trgm_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS analysis_runs_model_latest_idx ON analysis_runs (model_name, created_at DESC, id)")
    op.execute("CREATE INDEX IF NOT EXISTS embeddings_audio_aggregate_track_run_idx ON embeddings (track_id, run_id) WHERE embedding_type='audio-track' AND window_index IS NULL")
    op.execute("CREATE INDEX IF NOT EXISTS embeddings_lyrics_aggregate_track_run_idx ON embeddings (track_id, run_id) WHERE embedding_type='lyrics' AND window_index IS NULL")
    op.execute("CREATE INDEX IF NOT EXISTS library_tracks_track_idx ON library_tracks (track_id)")
    op.execute("CREATE INDEX IF NOT EXISTS retrieval_results_run_idx ON retrieval_results (run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS retrieval_results_track_idx ON retrieval_results (track_id)")
    op.execute("CREATE INDEX IF NOT EXISTS judgments_result_idx ON judgments (result_id)")
    op.execute("CREATE INDEX IF NOT EXISTS curations_user_updated_idx ON curations (user_id, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS community_snapshots_created_idx ON community_snapshots (created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS libraries_normalized_root_idx ON libraries (lower(rtrim(root_path, '/')))")
    op.execute("CREATE INDEX IF NOT EXISTS navidrome_connections_normalized_url_idx ON navidrome_connections (lower(rtrim(url, '/')))")


def downgrade() -> None:
    for name in (
        "navidrome_connections_normalized_url_idx", "libraries_normalized_root_idx",
        "community_snapshots_created_idx", "curations_user_updated_idx", "judgments_result_idx",
        "retrieval_results_track_idx", "retrieval_results_run_idx", "library_tracks_track_idx",
        "embeddings_lyrics_aggregate_track_run_idx", "embeddings_audio_aggregate_track_run_idx",
        "analysis_runs_model_latest_idx", "tracks_album_trgm_idx", "tracks_artist_trgm_idx",
        "tracks_title_trgm_idx",
    ):
        op.execute(f"DROP INDEX IF EXISTS {name}")
