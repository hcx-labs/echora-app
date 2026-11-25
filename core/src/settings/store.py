"""Queries on the instance's settings singletons: analysis settings and external processing.

``analysis_settings`` holds the model switches and the Hugging Face token;
``external_processing_settings`` holds the Modal configuration and deployment state. Every function takes an open connection; the caller owns the transaction.
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


def huggingface_token(db: Connection) -> dict[str, Any] | None:
    """The stored token, encrypted (``hf_token_encrypted``), and its ``hf_token_account``."""
    return fetch_one(
        db, "SELECT hf_token_encrypted, hf_token_account FROM analysis_settings WHERE singleton"
    )


def has_huggingface_token(db: Connection) -> bool:
    """Whether a Hugging Face token is stored (presence only)."""
    row = fetch_one(
        db,
        "SELECT hf_token_encrypted IS NOT NULL AS stored FROM analysis_settings WHERE singleton",
    )
    return bool(row and row["stored"])


def save_huggingface_token(db: Connection, encrypted: bytes | None, account: str | None) -> None:
    """Store the encrypted token and its account (``None`` clears them)."""
    db.execute(
        """INSERT INTO analysis_settings (singleton, hf_token_encrypted, hf_token_account, updated_at)
           VALUES (true, %s, %s, now()) ON CONFLICT (singleton) DO UPDATE
           SET hf_token_encrypted=EXCLUDED.hf_token_encrypted,
               hf_token_account=EXCLUDED.hf_token_account, updated_at=now()""",
        (encrypted, account),
    )


def external_processing(db: Connection) -> dict[str, Any] | None:
    """Every column of the external processing settings, or ``None`` before they are saved."""
    return fetch_one(db, "SELECT * FROM external_processing_settings WHERE singleton")


def save_external_processing(db: Connection, **values: Any) -> None:
    """Set the named external processing columns, creating the row first if needed."""
    db.execute(
        "INSERT INTO external_processing_settings (singleton) VALUES (true) ON CONFLICT DO NOTHING"
    )
    assignments = ", ".join(f"{name}=%s" for name in values)
    db.execute(
        f"UPDATE external_processing_settings SET {assignments} WHERE singleton",
        tuple(values.values()),
    )


def try_advisory_lock(db: Connection, key: int) -> bool:
    """Take a session advisory lock by its integer key if no one holds it."""
    row = fetch_one(db, "SELECT pg_try_advisory_lock(%s) AS locked", (key,))
    return bool(row and row["locked"])


def advisory_unlock(db: Connection, key: int) -> None:
    """Release a session advisory lock by its integer key."""
    db.execute("SELECT pg_advisory_unlock(%s)", (key,))


def latest_job(db: Connection, kind: str) -> dict[str, Any] | None:
    """The newest job of the kind: ``id``, ``status``, ``progress``, ``error``, ``created_at``."""
    return fetch_one(
        db,
        """SELECT id, status, progress, error, created_at FROM jobs WHERE kind=%s
                        ORDER BY created_at DESC LIMIT 1""",
        (kind,),
    )
