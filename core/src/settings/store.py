"""Queries on the instance's analysis settings (the ``analysis_settings`` singleton row).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

from psycopg import Connection

from core.db.connection import fetch_one


def transcription_enabled(db: Connection) -> bool:
    """Whether AI lyric generation is on anywhere (on this server or on Modal)."""
    row = fetch_one(
        db,
        """SELECT transcription_processing_enabled OR transcription_modal_enabled AS enabled
           FROM analysis_settings WHERE singleton=true""",
    )
    return bool(row and row["enabled"])
