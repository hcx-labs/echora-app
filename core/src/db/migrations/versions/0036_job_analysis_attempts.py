"""Associate representation attempts with the owning durable job."""
from alembic import op

revision = "0036_job_analysis_attempts"
down_revision = "0035_curation_publications"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE analysis_attempts ADD COLUMN job_id uuid REFERENCES jobs(id);\nCREATE INDEX analysis_attempts_job ON analysis_attempts(job_id) WHERE job_id IS NOT NULL;\n\nCREATE FUNCTION interrupt_job_analysis_attempts() RETURNS trigger LANGUAGE plpgsql AS $$\nBEGIN\n  IF OLD.status = 'running' AND NEW.status <> 'running' THEN\n    UPDATE analysis_attempts SET status='interrupted', finished_at=now()\n      WHERE job_id=NEW.id AND status='running';\n  END IF;\n  RETURN NEW;\nEND;\n$$;\nCREATE TRIGGER jobs_interrupt_attempts AFTER UPDATE OF status ON jobs\n  FOR EACH ROW EXECUTE FUNCTION interrupt_job_analysis_attempts();\n")


def downgrade() -> None:
    op.execute("DROP TRIGGER jobs_interrupt_attempts ON jobs")
    op.execute("DROP FUNCTION interrupt_job_analysis_attempts()")
    op.drop_column("analysis_attempts", "job_id")
