"""Store FA-Kara aligned lyrics.

Revision ID: 0008_karaoke_lyrics
Revises: 0007_oidc_auth
"""

from alembic import op

revision = "0008_karaoke_lyrics"
down_revision = "0007_oidc_auth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE lyrics ADD COLUMN IF NOT EXISTS karaoke_lines jsonb")
    op.execute("ALTER TABLE lyrics ADD COLUMN IF NOT EXISTS karaoke_ass text")
    op.execute("ALTER TABLE lyrics ADD COLUMN IF NOT EXISTS karaoke_lrc text")
    op.execute("ALTER TABLE lyrics ADD COLUMN IF NOT EXISTS karaoke_model text")
    op.execute("ALTER TABLE lyrics ADD COLUMN IF NOT EXISTS karaoke_model_revision text")
    op.execute("ALTER TABLE lyrics ADD COLUMN IF NOT EXISTS karaoke_created_at timestamptz")
    op.execute("CREATE INDEX IF NOT EXISTS lyrics_karaoke_pending_idx ON lyrics (track_id) WHERE text IS NOT NULL AND karaoke_lines IS NULL")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS lyrics_karaoke_pending_idx")
    for column in ("karaoke_created_at", "karaoke_model_revision", "karaoke_model", "karaoke_lrc", "karaoke_ass", "karaoke_lines"):
        op.execute(f"ALTER TABLE lyrics DROP COLUMN IF EXISTS {column}")
