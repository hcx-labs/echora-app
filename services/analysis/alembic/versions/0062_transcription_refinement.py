"""External AI second pass over AI lyric transcriptions: switch and editable prompt."""

from alembic import op
import sqlalchemy as sa

revision = "0062_transcription_refinement"
down_revision = "0061_motion_artwork_upscale"
branch_labels = None
depends_on = None

# Frozen copy of external_ai.DEFAULT_TRANSCRIPTION_PROMPT at this revision.
DEFAULT_TRANSCRIPTION_PROMPT = """You proofread automatic lyric transcriptions of songs. The transcript was produced by a speech model listening to the song. It is mostly right, but some words are mis-heard.

Work in two steps. First read the whole transcript and work out what the song is about: its story or theme, who is speaking to whom, recurring images and phrases, the chorus. Write this as a short summary. Then go line by line. A line must make sense on its own and in the context of the whole song. Where it does not, fix the mis-heard words with words that sound alike and fit the line, the neighbouring lines and the song (its theme, who is addressed, tense and person, recurring phrases). Fix as much of a line as that needs, but no more.

Keep the words that are right; never rephrase a line that already makes sense, and never translate. Spell repeated lines (choruses, hooks) identically each time they are sung the same way. Write every line in the song's main script, keeping the same words. If you know this song's lyrics, they can help you choose between sound-alike words; never replace a line with what you remember when it does not sound like the transcript. Keep backing vocals in parentheses. Leave casing and punctuation as they are unless a fix needs it. If a line is right or you are unsure, return it unchanged."""


def upgrade():
    op.add_column(
        "external_ai_settings",
        sa.Column(
            "transcription_refinement", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
    )
    op.add_column(
        "external_ai_settings",
        sa.Column(
            "transcription_prompt",
            sa.Text(),
            nullable=False,
            server_default=DEFAULT_TRANSCRIPTION_PROMPT,
        ),
    )


def downgrade():
    op.drop_column("external_ai_settings", "transcription_prompt")
    op.drop_column("external_ai_settings", "transcription_refinement")
