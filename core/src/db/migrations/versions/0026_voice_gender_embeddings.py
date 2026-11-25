"""Allow voice-gender classification vectors in embeddings.

Revision ID: 0026_voice_gender_embeddings
Revises: 0025_curation_prompt_tags
"""
from alembic import op

revision = "0026_voice_gender_embeddings"
down_revision = "0025_curation_prompt_tags"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE embeddings
      DROP CONSTRAINT IF EXISTS embeddings_embedding_type_check""")
    op.execute("""ALTER TABLE embeddings
      ADD CONSTRAINT embeddings_embedding_type_check
      CHECK (embedding_type = ANY (ARRAY['audio-track'::text, 'audio-window'::text, 'lyrics'::text, 'voice-gender'::text]))""")


def downgrade() -> None:
    op.execute("DELETE FROM embeddings WHERE embedding_type='voice-gender'")
    op.execute("""ALTER TABLE embeddings
      DROP CONSTRAINT IF EXISTS embeddings_embedding_type_check""")
    op.execute("""ALTER TABLE embeddings
      ADD CONSTRAINT embeddings_embedding_type_check
      CHECK (embedding_type = ANY (ARRAY['audio-track'::text, 'audio-window'::text, 'lyrics'::text]))""")
