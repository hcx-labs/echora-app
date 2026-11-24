"""External processing: Modal configuration, its confirmed state, and per-location feature switches."""

from alembic import op

revision = "0058_external_processing"
down_revision = "0057_motion_artwork"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE external_processing_settings (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            enabled boolean NOT NULL DEFAULT false,
            provider text NOT NULL DEFAULT 'modal' CHECK (provider IN ('modal')),
            -- Token IDs are not secret; the secret is Fernet-encrypted like other credentials.
            token_id text NOT NULL DEFAULT '',
            token_secret_encrypted bytea,
            gpu text NOT NULL DEFAULT 'L40S'
                CHECK (gpu IN ('T4', 'L4', 'A10G', 'L40S', 'A100-40GB', 'A100-80GB', 'H100')),
            -- Preselects "Process on Modal" when starting a sync.
            default_compute text NOT NULL DEFAULT 'local' CHECK (default_compute IN ('local', 'modal')),
            -- Non-admin users may choose Modal, spending the instance's Modal credit.
            allow_users boolean NOT NULL DEFAULT false,
            workspace text,
            -- What Echora last confirmed in the Modal workspace.
            status text NOT NULL DEFAULT 'unprepared'
                CHECK (status IN ('unprepared', 'preparing', 'ready', 'failed')),
            status_detail text,
            deployed_code text,
            deployed_models text,
            checked_at timestamptz,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        -- Optional features switch on per location. A sync runs on this server or on
        -- Modal and does only the features switched on there. Start Modal alike.
        ALTER TABLE analysis_settings
            ADD COLUMN transcription_modal_enabled boolean NOT NULL DEFAULT false,
            ADD COLUMN karaoke_modal_enabled boolean NOT NULL DEFAULT true,
            ADD COLUMN hum_modal_enabled boolean NOT NULL DEFAULT true;
        UPDATE analysis_settings SET transcription_modal_enabled=transcription_processing_enabled,
            karaoke_modal_enabled=karaoke_processing_enabled, hum_modal_enabled=hum_processing_enabled;
    """)


def downgrade():
    op.execute("""ALTER TABLE analysis_settings DROP COLUMN transcription_modal_enabled,
        DROP COLUMN karaoke_modal_enabled, DROP COLUMN hum_modal_enabled;
        DROP TABLE external_processing_settings""")
