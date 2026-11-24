"""Allow semantic fusion vectors in the embeddings table."""
from alembic import op

revision = "0037_semantic_fusion"
down_revision = "0036_job_analysis_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE embeddings DROP CONSTRAINT embeddings_embedding_type_check;\n"
        "ALTER TABLE embeddings ADD CONSTRAINT embeddings_embedding_type_check\n"
        "  CHECK (embedding_type = ANY (ARRAY['audio-track'::text, 'audio-window'::text,\n"
        "    'lyrics'::text, 'voice-gender'::text, 'semantic_fusion'::text]));"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE embeddings DROP CONSTRAINT embeddings_embedding_type_check;\n"
        "ALTER TABLE embeddings ADD CONSTRAINT embeddings_embedding_type_check\n"
        "  CHECK (embedding_type = ANY (ARRAY['audio-track'::text, 'audio-window'::text,\n"
        "    'lyrics'::text, 'voice-gender'::text]));"
    )
