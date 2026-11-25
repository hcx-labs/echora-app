"""Fill missing translations during sync; source lyrics remain authoritative."""

import functools
import importlib.util
import logging
import re
import uuid
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from core.db.config import require_database_url
from core.integrations.external_ai import LanguagePair
from core.library.connections import credential_cipher
from core.lyrics.translations import (
    TranslationError,
    load_settings,
    load_translations,
    save_translation,
    source_checksum,
)
from worker.translation.client import translate

logger = logging.getLogger(__name__)


def source_lines(row):
    timed = (row.get("provenance") or {}).get("lines") or []
    if timed and all(
        isinstance(line, dict) and isinstance(line.get("text"), str) for line in timed
    ):
        return [line["text"] for line in timed]
    return (row.get("text") or "").splitlines()


# Source language stored for a song in a target language with some lines in other languages.
MIXED_SOURCE = "mul"
# A song needs this many foreign lines before it is sent; fewer are usually vocables or names.
MIXED_MIN_LINES = 2
# fastText alone marks a line foreign only for a language translated from, and only this sure.
MIXED_LID_CONFIDENCE = 0.7


@functools.lru_cache(maxsize=1)
def _label_rules():
    """The aligner's per-line language rule (vendor/fa_kara/label_rules.py), loaded on first use."""
    path = Path(__file__).resolve().parents[1] / "kara" / "vendor" / "fa_kara" / "label_rules.py"
    spec = importlib.util.spec_from_file_location("echora_fa_kara_label_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def foreign_lines(lines: list[str], language: str, sources: set[str]) -> list[int]:
    """Indexes of lines not in the song's language: the karaoke aligner's per-line rule
    (romanised Urdu, Hindi, Korean, ... that fastText cannot name), or fastText confident in a
    language that is translated from (Spanish lines it names but the rule reads as English)."""
    rules = _label_rules()
    found = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if rules.line_language(line, language) != language:
            found.append(index)
            continue
        words = [word.lower() for word in rules._LID_WORD.findall(line)]
        if len(words) >= 3:
            top, confidence = rules.line_lid(" ".join(words))
            if top in sources and confidence >= MIXED_LID_CONFIDENCE:
                found.append(index)
    return found


def _comparable(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def keep_foreign(lines: list[str], translated: list[dict], foreign: list[int]) -> list[dict]:
    """Only translations of foreign lines, and only where the model changed them; every other line
    is left empty, which the player shows as no translation."""
    chosen = set(foreign)
    return [
        {
            "id": line["id"],
            "text": line["text"]
            if line["id"] in chosen and _comparable(line["text"]) != _comparable(source)
            else "",
        }
        for source, line in zip(lines, translated)
    ]


def backfill_translations(user_id, url, external_ids, *, check, progress):
    # `declined`: the model answered but would not translate (see lyric_translation._checked).
    summary = {"translated": 0, "reused": 0, "failed": 0, "declined": 0, "stale": 0}
    settings, encrypted = load_settings()
    if not settings.enabled or not settings.model or not settings.language_pairs:
        return summary

    key = credential_cipher().decrypt(bytes(encrypted)).decode() if encrypted else None
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
        work = [
            (pair, None)
            for pair in settings.language_pairs
            if language == pair.source
            or ("-" not in pair.source and language.split("-")[0] == pair.source)
        ]
        # A song already in a target language may still have lines in others (an Urdu verse in an
        # English song): translate those, with the whole song as context.
        if not work:
            base = language.split("-")[0]
            targets = sorted({p.target for p in settings.language_pairs if p.target == base})
            if targets:
                sources = {p.source.split("-")[0] for p in settings.language_pairs}
                try:
                    foreign = foreign_lines(lines, base, sources)
                except Exception:
                    logger.exception("Finding foreign lines failed for track %s", row["track_id"])
                    foreign = []
                if len(foreign) >= MIXED_MIN_LINES:
                    work = [
                        (LanguagePair(source=MIXED_SOURCE, target=target), foreign)
                        for target in targets
                    ]
        for pair, foreign in work:
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
                if foreign is None:
                    translated = translate(settings, key, pair, lines)
                else:
                    translated = keep_foreign(
                        lines, translate(settings, key, pair, lines, mixed=True), foreign
                    )
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
