"""External AI provider: an OpenAI-compatible endpoint or Claude through the Anthropic SDK."""

from alembic import op
import sqlalchemy as sa

revision = "0063_external_ai_provider"
down_revision = "0062_transcription_refinement"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "external_ai_settings",
        sa.Column("provider", sa.Text(), nullable=False, server_default="openai"),
    )
    op.create_check_constraint(
        "external_ai_settings_provider_check",
        "external_ai_settings",
        "provider IN ('openai', 'anthropic')",
    )


def downgrade():
    op.drop_constraint("external_ai_settings_provider_check", "external_ai_settings")
    op.drop_column("external_ai_settings", "provider")
