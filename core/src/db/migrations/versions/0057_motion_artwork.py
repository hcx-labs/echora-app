"""Generated motion artwork: settings, rendered loops and the track-to-cover mapping."""

from alembic import op

revision = "0057_motion_artwork"
down_revision = "0056_navidrome_original_lyrics"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE motion_artwork_settings (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            enabled boolean NOT NULL DEFAULT false,
            -- Blank starts the worker's embedded ComfyUI; an address uses a running one instead.
            comfyui_url text NOT NULL DEFAULT '',
            resolution integer NOT NULL DEFAULT 1536 CHECK (resolution IN (512, 768, 1024, 1536)),
            frames integer NOT NULL DEFAULT 121 CHECK (frames IN (97, 121)),
            prompt_mode text NOT NULL DEFAULT 'auto' CHECK (prompt_mode IN ('auto', 'external', 'fixed')),
            instructions text NOT NULL DEFAULT '',
            fixed_prompt text NOT NULL DEFAULT '',
            mid_anchor_strength real NOT NULL DEFAULT 0 CHECK (mid_anchor_strength BETWEEN 0 AND 1),
            seed bigint DEFAULT 42 CHECK (seed >= 0),
            -- Render missing loops in sync and import batches (in addition to explicit backfills).
            generate_during_sync boolean NOT NULL DEFAULT true,
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        -- One loop per cover image and recipe. Covers are identified by content, so albums
        -- sharing artwork share a loop and a replaced cover gets a new one.
        CREATE TABLE motion_artworks (
            id uuid PRIMARY KEY,
            cover_sha256 text NOT NULL,
            recipe_hash text NOT NULL,
            status text NOT NULL CHECK (status IN ('complete', 'failed')),
            prompt text,
            error text,
            path text,
            width integer,
            height integer,
            frames integer,
            fps integer,
            bytes bigint,
            recipe jsonb NOT NULL DEFAULT '{}',
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (cover_sha256, recipe_hash),
            CHECK (status <> 'complete' OR path IS NOT NULL)
        );
        CREATE INDEX motion_artworks_cover ON motion_artworks(cover_sha256, created_at DESC)
            WHERE status = 'complete';
        CREATE TABLE track_motion_artwork (
            track_id uuid PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
            cover_sha256 text NOT NULL,
            cover_art_id text,
            checked_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX track_motion_artwork_cover ON track_motion_artwork(cover_sha256);
    """)


def downgrade():
    # Rendered files under ECHORA_MOTION_ARTWORK_DIR are left on disk.
    op.execute("""DROP TABLE track_motion_artwork; DROP TABLE motion_artworks;
        DROP TABLE motion_artwork_settings""")
