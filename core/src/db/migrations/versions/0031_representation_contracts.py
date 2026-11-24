"""Select compatible representations and retain processing attempts."""
from alembic import op

revision = "0031_representation_contracts"
down_revision = "0030_composable_curations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""-- Deployment-selected contracts. Historical embeddings remain available for audit.
CREATE TABLE IF NOT EXISTS active_representation_specs (
  kind text NOT NULL,
  model_name text NOT NULL,
  model_revision text NOT NULL,
  config_hash text NOT NULL,
  dimension integer,
  config jsonb NOT NULL,
  PRIMARY KEY (kind, model_name)
);

CREATE OR REPLACE VIEW current_analysis_runs AS
SELECT ar.* FROM analysis_runs ar
JOIN active_representation_specs spec
  ON spec.kind=ar.kind AND spec.model_name=ar.model_name
 AND spec.model_revision=ar.model_revision AND spec.config_hash=ar.config_hash;

-- A committed embedding is complete per track. Batch status must not hide it
-- while another track in the same representation is being processed.
CREATE OR REPLACE VIEW current_embeddings AS
SELECT e.* FROM embeddings e
JOIN current_analysis_runs ar ON ar.id=e.run_id
JOIN active_representation_specs spec
  ON spec.kind=ar.kind AND spec.model_name=ar.model_name
WHERE e.dimension=spec.dimension;

CREATE OR REPLACE VIEW current_audio_profiles AS
SELECT tap.* FROM track_audio_profiles tap
JOIN current_analysis_runs source_run ON source_run.id=tap.source_run_id
JOIN current_analysis_runs profile_run ON profile_run.id=tap.profile_run_id
JOIN active_representation_specs spec
  ON spec.kind=source_run.kind AND spec.model_name=source_run.model_name
WHERE source_run.kind='audio_embedding' AND profile_run.kind='audio_profile'
  AND source_run.model_name=tap.model_name AND profile_run.model_name=tap.model_name
  AND tap.dimension=spec.dimension;

CREATE TABLE IF NOT EXISTS analysis_attempts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  status text NOT NULL DEFAULT 'running'
    CHECK (status IN ('running', 'complete', 'partial', 'failed', 'interrupted')),
  requested integer NOT NULL,
  succeeded integer NOT NULL DEFAULT 0,
  failed integer NOT NULL DEFAULT 0,
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz
);
CREATE TABLE IF NOT EXISTS analysis_attempt_tracks (
  attempt_id uuid NOT NULL REFERENCES analysis_attempts(id) ON DELETE CASCADE,
  external_id text NOT NULL,
  track_id uuid REFERENCES tracks(id) ON DELETE CASCADE,
  status text NOT NULL CHECK (status IN ('complete', 'failed')),
  error text,
  PRIMARY KEY (attempt_id, external_id)
);
""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS analysis_attempt_tracks")
    op.execute("DROP TABLE IF EXISTS analysis_attempts")
    op.execute("DROP VIEW IF EXISTS current_audio_profiles")
    op.execute("DROP VIEW IF EXISTS current_embeddings")
    op.execute("DROP VIEW IF EXISTS current_analysis_runs")
    op.execute("DROP TABLE IF EXISTS active_representation_specs")
