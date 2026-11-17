"""Fill missing translations during sync; source lyrics remain authoritative."""

import logging
import uuid

import psycopg
from psycopg.rows import dict_row

from .settings import require_database_url
from .translation_storage import load_settings, load_translations, save_translation
from .lyric_translation import translate, TranslationError, source_checksum

logger = logging.getLogger(__name__)


def source_lines(row):
    timed = (row.get("provenance") or {}).get("lines") or []
    if timed and all(
        isinstance(line, dict) and isinstance(line.get("text"), str) for line in timed
    ):
        return [line["text"] for line in timed]
    return (row.get("text") or "").splitlines()


def backfill_translations(user_id, url, external_ids, *, check, progress):
    # `declined`: the model answered but would not translate (see lyric_translation._checked).
    summary = {"translated": 0, "reused": 0, "failed": 0, "declined": 0, "stale": 0}
    settings, encrypted = load_settings()
    if not settings.enabled or not settings.model or not settings.language_pairs:
        return summary
    from . import main

    key = main._cipher().decrypt(bytes(encrypted)).decode() if encrypted else None
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/"))
    with psycopg.connect(require_database_url(), row_factory=dict_row) as db:
        tracks = db.execute(
            """SELECT DISTINCT l.track_id, l.text, l.language, l.provenance
            FROM lyrics l JOIN user_track_links u ON u.track_id=l.track_id
            JOIN libraries lib ON lib.id=u.library_id
            WHERE u.user_id=%s AND lib.namespace=%s AND u.external_id=ANY(%s)
              AND NULLIF(btrim(l.text),'') IS NOT NULL""",
            (user_id, namespace, external_ids),
        ).fetchall()
    for index, row in enumerate(tracks):
        check()
        lines = source_lines(row)
        language = (row["language"] or "").lower().replace("_", "-")
        for pair in settings.language_pairs:
            if language != pair.source and not (
                "-" not in pair.source and language.split("-")[0] == pair.source
            ):
                continue
            check()
            # Honor a disable action between requests, not just between batches.
            current, _ = load_settings()
            if not current.enabled:
                return summary
            cached = load_translations(row["track_id"], lines)
            if any(
                t["source_language"] == pair.source
                and t["target_language"] == pair.target
                and t["status"] == "ready"
                for t in cached
            ):
                summary["reused"] += 1
                continue
            try:
                translated = translate(settings, key, pair, lines)
                check()
                with psycopg.connect(require_database_url(), row_factory=dict_row) as db:
                    latest = db.execute(
                        "SELECT text, provenance FROM lyrics WHERE track_id=%s", (row["track_id"],)
                    ).fetchone()
                if latest is None or source_checksum(source_lines(latest)) != source_checksum(
                    lines
                ):
                    summary["stale"] += 1
                    continue
                save_translation(
                    row["track_id"],
                    pair,
                    lines,
                    translated,
                    model=settings.model,
                    prompt=settings.prompt,
                )
                summary["translated"] += 1
            except TranslationError as error:
                # No remote response bodies, keys, or lyrics in logs. A later sync retries.
                if str(error) in {"declined", "refusal"}:
                    logger.warning(
                        "Translation declined by the model for track %s (%s)",
                        row["track_id"],
                        error,
                    )
                    summary["declined"] += 1
                    continue
                logger.warning("Translation failed for track %s: %s", row["track_id"], error)
                summary["failed"] += 1
        progress(
            {
                "phase": "translation",
                "message": "Translating missing lyrics",
                "completed": index + 1,
                "total": len(tracks),
                "unit": "tracks",
                "summary": summary,
            }
        )
    return summary
