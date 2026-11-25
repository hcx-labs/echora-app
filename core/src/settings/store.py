"""Queries on the instance's analysis settings (the ``analysis_settings`` singleton row).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

from typing import Any

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


SWITCH_COLUMNS = frozenset(
    {
        "karaoke_processing_enabled",
        "karaoke_modal_enabled",
        "hum_processing_enabled",
        "hum_modal_enabled",
        "transcription_processing_enabled",
        "transcription_modal_enabled",
    }
)


def model_switches(db: Connection) -> dict[str, Any] | None:
    """The feature switches and the stored Hugging Face account (``None`` before any save).

    ``has_huggingface_token`` says whether a token is stored, never the token.
    """
    return fetch_one(
        db,
        """SELECT karaoke_processing_enabled, hum_processing_enabled,
                  transcription_processing_enabled, karaoke_modal_enabled,
                  hum_modal_enabled, transcription_modal_enabled,
                  hf_token_account, hf_token_encrypted IS NOT NULL AS has_huggingface_token
           FROM analysis_settings WHERE singleton=true""",
    )


def set_switch(db: Connection, column: str, enabled: bool, *, unbind_karaoke: bool) -> None:
    """Turn one feature switch (``SWITCH_COLUMNS``) on or off.

    ``unbind_karaoke`` also clears ``karaoke_bound_to_synced_lines``, as the karaoke and hum
    switches do.
    """
    if column not in SWITCH_COLUMNS:
        raise ValueError(f"Unknown feature switch {column!r}")
    if unbind_karaoke:
        db.execute(
            f"""INSERT INTO analysis_settings
                 (singleton, {column}, karaoke_bound_to_synced_lines, updated_at)
               VALUES (true,%s,false,now()) ON CONFLICT (singleton) DO UPDATE
               SET {column}=EXCLUDED.{column},
                   karaoke_bound_to_synced_lines=false, updated_at=now()""",
            (enabled,),
        )
    else:
        db.execute(
            f"""INSERT INTO analysis_settings (singleton, {column}, updated_at)
               VALUES (true,%s,now()) ON CONFLICT (singleton) DO UPDATE
               SET {column}=EXCLUDED.{column}, updated_at=now()""",
            (enabled,),
        )
