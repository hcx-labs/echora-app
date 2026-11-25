"""Add the visual-feature availability status to databases created before revision 0042."""
from alembic import op

revision = "0043_visual_feature_status"
down_revision = "0042_track_visual_features"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE track_visual_features
        ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'complete'
        CHECK (status IN ('complete', 'unsupported'))
    """)


def downgrade():
    op.drop_column("track_visual_features", "status")
