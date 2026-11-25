"""Motion artwork pinned to the cover at both ends, with optional two-stage upscaling.

Loops now start and end on the cover instead of playing forward and back, so the partial
turnaround keyframe is gone. Rendering moves to the new defaults (768 upscaled to 1536,
10 seconds). Existing loops are kept: syncs render loops made with other settings again only
when regenerate_outdated is on.
"""

from alembic import op

revision = "0061_motion_artwork_upscale"
down_revision = "0060_huggingface_token"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE motion_artwork_settings
            ADD COLUMN upscale boolean NOT NULL DEFAULT true,
            ADD COLUMN regenerate_outdated boolean NOT NULL DEFAULT false,
            ADD COLUMN camera_lock real NOT NULL DEFAULT 0.5 CHECK (camera_lock BETWEEN 0 AND 1),
            DROP COLUMN mid_anchor_strength,
            DROP CONSTRAINT motion_artwork_settings_frames_check,
            ALTER COLUMN resolution SET DEFAULT 768,
            ALTER COLUMN frames SET DEFAULT 241;
        UPDATE motion_artwork_settings SET resolution = 768, frames = 241, upscale = true;
        ALTER TABLE motion_artwork_settings
            ADD CONSTRAINT motion_artwork_settings_frames_check CHECK (frames IN (121, 241)),
            ADD CONSTRAINT motion_artwork_settings_upscale_check CHECK (NOT upscale OR resolution <= 768);
    """)


def downgrade():
    op.execute("""
        ALTER TABLE motion_artwork_settings
            DROP CONSTRAINT motion_artwork_settings_upscale_check,
            DROP CONSTRAINT motion_artwork_settings_frames_check,
            DROP COLUMN upscale,
            DROP COLUMN regenerate_outdated,
            DROP COLUMN camera_lock,
            ADD COLUMN mid_anchor_strength real NOT NULL DEFAULT 0
                CHECK (mid_anchor_strength BETWEEN 0 AND 1),
            ALTER COLUMN resolution SET DEFAULT 1536,
            ALTER COLUMN frames SET DEFAULT 121;
        UPDATE motion_artwork_settings SET frames = 121;
        ALTER TABLE motion_artwork_settings
            ADD CONSTRAINT motion_artwork_settings_frames_check CHECK (frames IN (97, 121));
    """)
