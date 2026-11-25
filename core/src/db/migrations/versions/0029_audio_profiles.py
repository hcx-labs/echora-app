"""Add versioned multi-vector audio profiles.

Revision ID: 0029_audio_profiles
Revises: 0028_refresh_interval_24h
"""
from alembic import op

revision = "0029_audio_profiles"
down_revision = "0028_refresh_interval_24h"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE IF NOT EXISTS track_audio_profiles (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
      profile_run_id uuid NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
      source_run_id uuid NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
      model_name text NOT NULL CHECK (model_name IN ('muq_mulan', 'mert')),
      dimension integer NOT NULL CHECK (dimension > 0),
      global_overlap_embedding vector NOT NULL,
      global_decorrelated_embedding vector NOT NULL,
      resultant_length double precision NOT NULL,
      mean_global_similarity double precision NOT NULL,
      p05_global_similarity double precision NOT NULL,
      adjacent_change_mean double precision NOT NULL,
      adjacent_change_p95 double precision NOT NULL,
      timestamps_exact boolean NOT NULL DEFAULT false,
      mode_count integer NOT NULL CHECK (mode_count > 0),
      created_at timestamptz NOT NULL DEFAULT now(),
      UNIQUE (track_id, profile_run_id, source_run_id)
    )""")
    op.execute("""CREATE INDEX IF NOT EXISTS track_audio_profiles_source_idx
      ON track_audio_profiles (source_run_id, track_id)""")
    op.execute("""CREATE TABLE IF NOT EXISTS audio_temporal_segments (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      profile_id uuid NOT NULL REFERENCES track_audio_profiles(id) ON DELETE CASCADE,
      segment_index integer NOT NULL CHECK (segment_index >= 0),
      start_seconds double precision NOT NULL CHECK (start_seconds >= 0),
      end_seconds double precision NOT NULL,
      dimension integer NOT NULL CHECK (dimension > 0),
      embedding vector NOT NULL,
      cohesion double precision NOT NULL,
      representative_window_index integer NOT NULL CHECK (representative_window_index >= 0),
      created_at timestamptz NOT NULL DEFAULT now(),
      CHECK (end_seconds >= start_seconds),
      UNIQUE (profile_id, segment_index)
    )""")
    op.execute("""CREATE INDEX IF NOT EXISTS audio_temporal_segments_track_idx
      ON audio_temporal_segments (profile_id, segment_index)""")
    op.execute("""CREATE TABLE IF NOT EXISTS audio_modes (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      profile_id uuid NOT NULL REFERENCES track_audio_profiles(id) ON DELETE CASCADE,
      mode_index integer NOT NULL CHECK (mode_index >= 0),
      dimension integer NOT NULL CHECK (dimension > 0),
      embedding vector NOT NULL,
      duration_weight double precision NOT NULL CHECK (duration_weight > 0 AND duration_weight <= 1),
      cohesion double precision NOT NULL,
      representative_window_index integer NOT NULL CHECK (representative_window_index >= 0),
      created_at timestamptz NOT NULL DEFAULT now(),
      UNIQUE (profile_id, mode_index)
    )""")
    op.execute("""CREATE INDEX IF NOT EXISTS audio_modes_track_idx
      ON audio_modes (profile_id, mode_index)""")
    op.execute("""CREATE TABLE IF NOT EXISTS audio_mode_intervals (
      mode_id uuid NOT NULL REFERENCES audio_modes(id) ON DELETE CASCADE,
      interval_index integer NOT NULL CHECK (interval_index >= 0),
      start_seconds double precision NOT NULL CHECK (start_seconds >= 0),
      end_seconds double precision NOT NULL,
      CHECK (end_seconds >= start_seconds),
      PRIMARY KEY (mode_id, interval_index)
    )""")


def downgrade() -> None:
    op.drop_table("audio_mode_intervals")
    op.drop_index("audio_modes_track_idx", table_name="audio_modes")
    op.drop_table("audio_modes")
    op.drop_index("audio_temporal_segments_track_idx", table_name="audio_temporal_segments")
    op.drop_table("audio_temporal_segments")
    op.drop_index("track_audio_profiles_source_idx", table_name="track_audio_profiles")
    op.drop_table("track_audio_profiles")
