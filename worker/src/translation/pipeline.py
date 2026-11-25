"""Fill missing translations during sync; source lyrics remain authoritative."""

from __future__ import annotations

import logging
import re
import uuid

from core.db.connection import connect
from core.integrations.external_ai import LanguagePair
from core.library.connections import credential_cipher
from core.lyrics.translations import (
    TranslationError,
    load_settings,
    load_translations,
    save_translation,
    source_checksum,
)
from worker.kara.aligner import label_rules
from worker.processing import Item, ItemResult, ProcessingContext, Processor
from worker.translation.client import translate
from worker.translation.errors import TranslationFailedError

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


def foreign_lines(lines: list[str], language: str, sources: set[str]) -> list[int]:
    """Indexes of lines not in the song's language.

    Either the karaoke aligner's per-line rule (romanised Urdu, Hindi, Korean, ... that fastText
    cannot name), or fastText confident in a language that is translated from (Spanish lines it
    names but the rule reads as English).
    """
    rules = label_rules
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
    """Only translations of foreign lines, and only where the model changed them.

    Every other line is left empty, which the player shows as no translation.
    """
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


class TranslationItem(Item):
    """One song and one language pair to translate it for."""

    track_id: uuid.UUID
    lines: tuple[str, ...]
    pair: LanguagePair
    foreign: tuple[int, ...] | None = None
    """For a song already in the target language: the lines in other languages, else None."""


class TranslationProcessor(Processor[None, TranslationItem, list[dict]]):
    """Fills missing translations of the batch's lyrics for the configured language pairs.

    Source lyrics stay authoritative: a translation whose lyrics changed while it was being
    written is dropped (``stale``). ``declined`` counts songs the model would not translate.
    """

    name = "translation"
    summary_key = "translations"
    counters = ("translated", "reused", "failed", "declined", "stale")
    total_key = None
    activity = "Translating missing lyrics"

    def plan(self, context: ProcessingContext) -> list[TranslationItem]:
        self.settings, encrypted = load_settings()
        settings = self.settings
        if not settings.enabled or not settings.model or not settings.language_pairs:
            return []
        self.key = credential_cipher().decrypt(bytes(encrypted)).decode() if encrypted else None
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, context.library_url.rstrip("/"))
        with connect(dict_rows=True) as db:
            tracks = db.execute(
                """SELECT DISTINCT l.track_id, l.text, l.language, l.provenance
                FROM lyrics l JOIN user_track_links u ON u.track_id=l.track_id
                JOIN libraries lib ON lib.id=u.library_id
                WHERE u.user_id=%s AND lib.namespace=%s AND u.external_id=ANY(%s)
                  AND NULLIF(btrim(l.text),'') IS NOT NULL""",
                (context.user_id, namespace, list(context.track_ids)),
            ).fetchall()
        items = []
        for row in tracks:
            context.check()
            lines = tuple(source_lines(row))
            items += [
                TranslationItem(
                    key=str(row["track_id"]),
                    track_id=row["track_id"],
                    lines=lines,
                    pair=pair,
                    foreign=foreign,
                )
                for pair, foreign in self._work(row, lines)
            ]
        return items

    def _work(self, row, lines) -> list[tuple[LanguagePair, tuple[int, ...] | None]]:
        pairs = self.settings.language_pairs
        language = (row["language"] or "").lower().replace("_", "-")
        work = [
            (pair, None)
            for pair in pairs
            if language == pair.source
            or ("-" not in pair.source and language.split("-")[0] == pair.source)
        ]
        if work:
            return work
        # A song already in a target language may still have lines in others (an Urdu verse in
        # an English song): translate those, with the whole song as context.
        base = language.split("-")[0]
        targets = sorted({p.target for p in pairs if p.target == base})
        if not targets:
            return []
        sources = {p.source.split("-")[0] for p in pairs}
        try:
            foreign = foreign_lines(list(lines), base, sources)
        except Exception as error:  # noqa: BLE001 - optional; the song keeps its lyrics
            logger.error(
                "Finding foreign lines failed for track %s: %s",
                row["track_id"],
                type(error).__name__,
            )
            return []
        if len(foreign) < MIXED_MIN_LINES:
            return []
        return [
            (LanguagePair(source=MIXED_SOURCE, target=target), tuple(foreign)) for target in targets
        ]

    def process(self, context, items):
        for item in items:
            context.check()
            # Honor a disable action between requests, not just between batches.
            if not load_settings()[0].enabled:
                return
            pair, lines = item.pair, list(item.lines)
            if any(
                t["source_language"] == pair.source
                and t["target_language"] == pair.target
                and t["status"] == "ready"
                for t in load_translations(item.track_id, lines)
            ):
                yield ItemResult(item=item, outcome="reused")
                continue
            try:
                if item.foreign is None:
                    translated = translate(self.settings, self.key, pair, lines)
                else:
                    translated = keep_foreign(
                        lines,
                        translate(self.settings, self.key, pair, lines, mixed=True),
                        list(item.foreign),
                    )
            except TranslationError as error:
                # Only the error code: no response bodies, keys or lyrics. A later sync retries.
                if str(error) in {"declined", "refusal"}:
                    logger.warning(
                        "Translation declined by the model for track %s (%s)",
                        item.track_id,
                        error,
                    )
                    yield ItemResult(item=item, outcome="declined")
                else:
                    failure = TranslationFailedError("Translation failed", stage=str(error))
                    yield ItemResult.failed(item, failure)
                continue
            context.check()
            if self._lyrics_changed(item):
                yield ItemResult(item=item, outcome="stale")
                continue
            yield ItemResult(item=item, outcome="translated", output=translated)

    @staticmethod
    def _lyrics_changed(item: TranslationItem) -> bool:
        with connect(dict_rows=True) as db:
            latest = db.execute(
                "SELECT text, provenance FROM lyrics WHERE track_id=%s", (item.track_id,)
            ).fetchone()
        return latest is None or source_checksum(source_lines(latest)) != source_checksum(
            list(item.lines)
        )

    def persist(self, context, item, output):
        save_translation(
            item.track_id,
            item.pair,
            list(item.lines),
            output,
            model=self.settings.model,
            prompt=self.settings.prompt,
        )
