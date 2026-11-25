"""Motion artwork in syncs on Modal: its own switch, beside generating in syncs on this server."""

from alembic import op

revision = "0059_motion_artwork_on_modal"
down_revision = "0058_external_processing"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE motion_artwork_settings ADD COLUMN generate_on_modal boolean NOT NULL DEFAULT false"
    )


def downgrade():
    op.execute("ALTER TABLE motion_artwork_settings DROP COLUMN generate_on_modal")
