"""Add language targeting columns to curations.

Revision ID: 0027_curation_language_target
Revises: 0026_voice_gender_embeddings
"""
import sqlalchemy as sa
from alembic import op

revision = "0027_curation_language_target"
down_revision = "0026_voice_gender_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("curations", sa.Column("target_language", sa.Text(), nullable=False, server_default="", existing_type=sa.Text()))
    op.add_column("curations", sa.Column("language_strictness", sa.Text(), nullable=False, server_default="primarily", existing_type=sa.Text()))


def downgrade() -> None:
    op.drop_column("curations", "language_strictness")
    op.drop_column("curations", "target_language")
