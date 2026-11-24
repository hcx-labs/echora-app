"""Add canonical karaoke alignment documents and diagnostics.

Revision ID: 0012_karaoke_documents
Revises: 0011_anchored_karaoke
"""
from alembic import op

revision = "0012_karaoke_documents"
down_revision = "0011_anchored_karaoke"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""ALTER TABLE karaoke_lyrics_variants
      ADD COLUMN IF NOT EXISTS alignment_document JSONB,
      ADD COLUMN IF NOT EXISTS diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb""")
    op.execute("""ALTER TABLE karaoke_lyrics_variants
      DROP CONSTRAINT IF EXISTS karaoke_alignment_document_schema""")
    op.execute("""ALTER TABLE karaoke_lyrics_variants
      ADD CONSTRAINT karaoke_alignment_document_schema CHECK (
        alignment_document IS NULL OR alignment_document->>'schema_version' = '1'
      )""")


def downgrade() -> None:
    op.execute("ALTER TABLE karaoke_lyrics_variants DROP COLUMN IF EXISTS diagnostics")
    op.execute("ALTER TABLE karaoke_lyrics_variants DROP COLUMN IF EXISTS alignment_document")
