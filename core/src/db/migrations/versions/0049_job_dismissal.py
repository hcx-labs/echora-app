"""Remember job-result dismissal for the job owner."""
from alembic import op

revision = "0049_job_dismissal"
down_revision = "0048_source_freshness"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE jobs ADD COLUMN dismissed_at timestamptz")


def downgrade():
    op.execute("ALTER TABLE jobs DROP COLUMN dismissed_at")
