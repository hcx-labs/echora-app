"""External AI configuration and separate line-indexed translations."""
from alembic import op

revision = "0051_external_ai_translations"
down_revision = "0050_session_activity"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE external_ai_settings (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            enabled boolean NOT NULL DEFAULT false,
            url text NOT NULL DEFAULT '', model text NOT NULL DEFAULT '',
            prompt text NOT NULL,
            language_pairs jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(language_pairs) = 'array'),
            api_key_encrypted bytea,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE lyric_translations (
            track_id uuid NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
            source_language text NOT NULL,
            target_language text NOT NULL CHECK (target_language <> source_language),
            provenance text NOT NULL CHECK (provenance IN ('ai', 'provider')),
            lines jsonb NOT NULL CHECK (jsonb_typeof(lines) = 'array'),
            source_checksum text NOT NULL CHECK (length(source_checksum) = 64),
            prompt_revision text NOT NULL,
            model text NOT NULL,
            status text NOT NULL DEFAULT 'ready' CHECK (status IN ('ready', 'stale', 'failed')),
            error_code text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            failed_at timestamptz,
            PRIMARY KEY (track_id, source_language, target_language, provenance)
        );
    """)


def downgrade():
    op.execute("DROP TABLE lyric_translations; DROP TABLE external_ai_settings")
