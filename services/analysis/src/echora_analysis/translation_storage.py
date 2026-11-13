"""PostgreSQL persistence. Original lyrics are never written by this module."""

import hashlib

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .external_ai import ExternalAISettings, LanguagePair
from .lyric_translation import source_checksum, validate_lines
from .settings import require_database_url


def load_settings():
    with psycopg.connect(require_database_url(), row_factory=dict_row) as db:
        row = db.execute("SELECT * FROM external_ai_settings WHERE singleton").fetchone()
    if row is None:
        return ExternalAISettings(), None
    return ExternalAISettings.model_validate(
        {k: row[k] for k in ExternalAISettings.model_fields}
    ), row["api_key_encrypted"]


def save_settings(settings, encrypted_key, *, replace_key):
    with psycopg.connect(require_database_url()) as db:
        row = db.execute(
            """
            INSERT INTO external_ai_settings (singleton, enabled, url, model, prompt, language_pairs, api_key_encrypted)
            VALUES (true, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (singleton) DO UPDATE SET enabled=EXCLUDED.enabled,
                url=EXCLUDED.url, model=EXCLUDED.model, prompt=EXCLUDED.prompt,
                language_pairs=EXCLUDED.language_pairs, updated_at=now(),
                api_key_encrypted=CASE WHEN %s THEN EXCLUDED.api_key_encrypted ELSE external_ai_settings.api_key_encrypted END
            RETURNING api_key_encrypted IS NOT NULL
        """,
            (
                settings.enabled,
                settings.url,
                settings.model,
                settings.prompt,
                Jsonb([p.model_dump() for p in settings.language_pairs]),
                encrypted_key,
                replace_key,
            ),
        ).fetchone()
    return row[0]


def save_translation(
    track_id,
    pair: LanguagePair,
    source_lines: list[str],
    translated_lines,
    *,
    model: str,
    prompt: str,
    provenance: str = "ai",
):
    if provenance not in {"ai", "provider"}:
        raise ValueError("Invalid provenance")
    lines = validate_lines(translated_lines, len(source_lines))
    with psycopg.connect(require_database_url()) as db:
        db.execute(
            """
            INSERT INTO lyric_translations
                (track_id, source_language, target_language, provenance, lines, source_checksum, prompt_revision, model)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (track_id, source_language, target_language, provenance) DO UPDATE SET
                lines=EXCLUDED.lines, source_checksum=EXCLUDED.source_checksum,
                prompt_revision=EXCLUDED.prompt_revision, model=EXCLUDED.model,
                status='ready', error_code=NULL, failed_at=NULL, updated_at=now()
        """,
            (
                track_id,
                pair.source,
                pair.target,
                provenance,
                Jsonb(lines),
                source_checksum(source_lines),
                hashlib.sha256(prompt.encode()).hexdigest(),
                model,
            ),
        )


def load_translations(track_id, source_lines: list[str]):
    """Refresh staleness against resolved source text before exposing translations."""
    checksum = source_checksum(source_lines)
    with psycopg.connect(require_database_url(), row_factory=dict_row) as db:
        db.execute(
            """UPDATE lyric_translations SET status='stale', updated_at=now()
                      WHERE track_id=%s AND source_checksum<>%s AND status='ready'""",
            (track_id, checksum),
        )
        return db.execute(
            "SELECT * FROM lyric_translations WHERE track_id=%s ORDER BY source_language, target_language, provenance",
            (track_id,),
        ).fetchall()


def save_section(values, encrypted_key=None, *, replace_key=False):
    """Update only submitted section columns, inside one transaction."""
    from psycopg import sql
    from .external_ai import DEFAULT_PROMPT

    allowed = {"enabled", "url", "model", "prompt", "language_pairs"}
    if not values or not set(values) <= allowed:
        raise ValueError("Invalid settings fields")
    values = dict(values)
    if "language_pairs" in values:
        values["language_pairs"] = Jsonb(values["language_pairs"])
    if replace_key:
        values["api_key_encrypted"] = encrypted_key
    with psycopg.connect(require_database_url(), row_factory=dict_row) as db:
        db.execute(
            "INSERT INTO external_ai_settings (prompt) VALUES (%s) ON CONFLICT DO NOTHING",
            (DEFAULT_PROMPT,),
        )
        assignments = sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in values)
        row = db.execute(
            sql.SQL(
                "UPDATE external_ai_settings SET {}, updated_at=now() WHERE singleton RETURNING *"
            ).format(assignments),
            list(values.values()),
        ).fetchone()
    settings = ExternalAISettings.model_validate(
        {k: row[k] for k in ExternalAISettings.model_fields}
    )
    return {**settings.model_dump(), "has_key": bool(row["api_key_encrypted"])}


def clear_translations():
    """Delete translations only. Active job completion must precede deletion."""
    with psycopg.connect(require_database_url()) as db:
        # Block new job inserts until the check and deletion commit together.
        db.execute("LOCK TABLE jobs IN SHARE ROW EXCLUSIVE MODE")
        active = db.execute("""SELECT 1 FROM jobs WHERE status IN ('queued','running','waiting')
            AND (kind IN ('navidrome_sync','import','lyrics_backfill')
                 OR (kind='analysis_batch' AND payload->>'operation' IN ('navidrome_sync','import','lyrics_backfill')))
            LIMIT 1""").fetchone()
        if active:
            raise RuntimeError("translation_jobs_active")
        return db.execute("DELETE FROM lyric_translations").rowcount
