"""Song theme descriptions written by External AI, embedded for curation's Themes search."""

from alembic import op
import sqlalchemy as sa

revision = "0064_lyric_themes"
down_revision = "0063_external_ai_provider"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "external_ai_settings",
        sa.Column("lyric_themes", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # One row per track. `source_checksum` is sha256 of lyrics.text, so curation can tell a
    # description written for older lyrics; `prompt_revision` does the same for the prompt.
    op.execute(
        """CREATE TABLE track_lyric_themes (
             track_id uuid PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
             source_checksum text NOT NULL,
             prompt_revision text NOT NULL,
             model text NOT NULL,
             status text NOT NULL CHECK (status IN ('ready', 'failed')),
             error_code text,
             description text,
             themes jsonb NOT NULL DEFAULT '[]'::jsonb,
             embedding vector(1024),
             embedding_model text,
             updated_at timestamptz NOT NULL DEFAULT now(),
             CHECK (status <> 'ready' OR (description IS NOT NULL AND embedding IS NOT NULL))
           )"""
    )


def downgrade():
    op.execute("DROP TABLE track_lyric_themes")
    op.drop_column("external_ai_settings", "lyric_themes")
