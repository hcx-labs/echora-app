"""Allow multiple melody sources per track.

Revision ID: 0020_multiple_melody_sources
Revises: 0019_melody_contours
"""
from alembic import op

revision = "0020_multiple_melody_sources"
down_revision = "0019_melody_contours"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE melody_contours DROP CONSTRAINT melody_contours_source_check")
    op.execute("""ALTER TABLE melody_contours ADD CONSTRAINT melody_contours_source_check
                CHECK (source IN ('predominant-melody','full-mix','vocals','accompaniment','hum-query'))""")


def downgrade() -> None:
    op.execute("DELETE FROM melody_contours WHERE source IN ('full-mix','vocals','accompaniment')")
    op.execute("ALTER TABLE melody_contours DROP CONSTRAINT melody_contours_source_check")
    op.execute("""ALTER TABLE melody_contours ADD CONSTRAINT melody_contours_source_check
                CHECK (source IN ('predominant-melody','hum-query'))""")
