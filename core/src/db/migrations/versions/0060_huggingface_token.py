"""One Hugging Face token for gated model downloads, on this server and on Modal.

Lives with the analysis settings. Databases that ran an earlier 0058 kept it on the
External processing settings; that column is dropped (it was never released).
"""

from alembic import op

revision = "0060_huggingface_token"
down_revision = "0059_motion_artwork_on_modal"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE analysis_settings
            ADD COLUMN hf_token_encrypted bytea,
            -- The account the token belonged to when saved, shown in settings.
            ADD COLUMN hf_token_account text;
        ALTER TABLE external_processing_settings DROP COLUMN IF EXISTS hf_token_encrypted;
    """)


def downgrade():
    op.execute(
        "ALTER TABLE analysis_settings DROP COLUMN hf_token_encrypted, DROP COLUMN hf_token_account"
    )
