"""Track source verification independently of canonical artifact completeness."""
from alembic import op
import sqlalchemy as sa

revision = "0048_source_freshness"
down_revision = "0047_source_membership"
branch_labels = None
depends_on = None


def upgrade():
    # NULL deliberately means unverified, including pre-migration sources.
    op.add_column("track_sources", sa.Column("audio_verified_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column("track_sources", "audio_verified_at")
