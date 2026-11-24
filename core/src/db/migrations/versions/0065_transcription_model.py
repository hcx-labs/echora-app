"""Transcript refinement can use its own External AI model; empty uses the shared one."""

from alembic import op
import sqlalchemy as sa

revision = "0065_transcription_model"
down_revision = "0064_lyric_themes"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "external_ai_settings",
        sa.Column("transcription_model", sa.Text(), nullable=False, server_default=""),
    )


def downgrade():
    op.drop_column("external_ai_settings", "transcription_model")
