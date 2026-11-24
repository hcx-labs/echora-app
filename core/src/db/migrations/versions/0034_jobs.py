"""Durable fenced analysis jobs."""
from alembic import op

revision = "0034_jobs"
down_revision = "0033_track_waveforms"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE jobs (
  id uuid PRIMARY KEY,
  kind text NOT NULL,
  worker_type text NOT NULL,
  user_id uuid NOT NULL,
  connection_id uuid,
  parent_id uuid REFERENCES jobs(id),
  payload jsonb NOT NULL DEFAULT '{}',
  dedupe_key text,
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','waiting','complete','partial','failed','cancelled')),
  cancel_requested boolean NOT NULL DEFAULT false,
  progress jsonb NOT NULL DEFAULT '{}',
  summary jsonb,
  error text,
  attempts integer NOT NULL DEFAULT 0,
  max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
  available_at timestamptz NOT NULL DEFAULT now(),
  lease_until timestamptz,
  token uuid,
  worker_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz
);
CREATE UNIQUE INDEX jobs_active_dedupe ON jobs (user_id, dedupe_key)
  WHERE dedupe_key IS NOT NULL AND status IN ('queued','running','waiting');
CREATE INDEX jobs_claim ON jobs (worker_type, available_at, created_at)
  WHERE status IN ('queued','running');
CREATE INDEX jobs_owner ON jobs (user_id, created_at DESC);
CREATE INDEX jobs_parent ON jobs (parent_id) WHERE parent_id IS NOT NULL;
""")


def downgrade() -> None:
    op.drop_table("jobs")
