"""The lyrics stage: retrieval or transcription, refinement and lyric embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import uuid
from contextlib import ExitStack
from dataclasses import dataclass

import psycopg
import torch
from psycopg.types.json import Jsonb

from core.db.connection import connect
from core.embeddings.audio_models import release_model
from core.embeddings.config import EmbeddingConfig
from core.embeddings.representations import configure_representations, embedding_config
from core.embeddings.text import LyricsEmbeddingModel
from core.errors import ProcessingError
from core.integrations.navidrome import NavidromeClient
from core.library.connections import credential_cipher
from core.lyrics.language import detect_distribution
from core.lyrics.sources import store_original
from worker.library.attempts import finish_attempt, record_track, start_attempt
from worker.media.preprocessing import prepare_audio
from worker.media.roformer import separation_phase
from worker.planning.audio import plan_lyrics, resolve_library_id
from worker.processing import Item, ItemResult, ProcessingContext, Processor, run_stage
from worker.remote.session import RemoteModel
from worker.transcription.recovery import diagnostic_writer
from worker.transcription.song import SongTranscriber, needs_repair

logger = logging.getLogger(__name__)


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _create_run(connection: psycopg.Connection, model: LyricsEmbeddingModel) -> uuid.UUID:
    config = embedding_config(model.name, model.revision)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO analysis_runs
                 (kind, model_name, model_revision, config_hash, config, environment, device, precision, status, started_at)
               VALUES ('lyrics_embedding',%s,%s,%s,%s,%s,%s,'float32','running',now())
               ON CONFLICT (kind, model_name, model_revision, config_hash)
               DO UPDATE SET id=analysis_runs.id RETURNING id""",
            (
                model.name,
                model.revision,
                config_hash,
                Jsonb(config),
                Jsonb(environment),
                str(model.device),
            ),
        )
        return cursor.fetchone()[0]


def word_timing_digest(lines) -> str:
    return hashlib.sha256(
        json.dumps(lines, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def _provenance(result: dict[str, object], language_distribution: dict) -> dict[str, object]:
    return {
        "provider": "moss" if result.get("ai_generated") else "navidrome",
        "endpoint": result.get("source"),
        "ai_generated": bool(result.get("ai_generated")),
        **({"transcription": result["transcription"]} if result.get("transcription") else {}),
        "synced": result.get("synced"),
        "lines": result.get("lines") or [],
        **({"languages": language_distribution} if language_distribution else {}),
        # Word timing supplied with the lyrics: the karaoke step stores it instead of aligning.
        **(
            {
                "word_timing": result["word_timing"],
                "word_timing_digest": word_timing_digest(result["word_timing"]),
            }
            if result.get("word_timing") and not result.get("ai_generated")
            else {}
        ),
        **(
            {"instrumental_evidence": result["instrumental_evidence"]}
            if result.get("instrumental_evidence")
            else {}
        ),
    }


def _instrumental_evidence(connection, track_id) -> dict | None:
    from worker.transcription.evidence import instrumental_song

    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT a.activity FROM track_vocal_activity a
               JOIN current_embeddings e ON e.track_id=a.track_id AND e.run_id=a.run_id
               WHERE a.track_id=%s AND e.embedding_type='voice-gender'
               ORDER BY a.created_at DESC LIMIT 1""",
            (track_id,),
        )
        row = cursor.fetchone()
    return instrumental_song(row[0]) if row else None


def _language(language_distribution: dict, result: dict[str, object]) -> str | None:
    language = language_distribution.get("primary_language") or result.get("language")
    return None if language == "xxx" else language


def _store_lyrics(
    connection: psycopg.Connection, track_id: uuid.UUID, result: dict[str, object]
) -> uuid.UUID | None:
    text = result.get("text")
    status = str(result.get("status") or "unavailable")
    source = "transcribed" if result.get("ai_generated") else "embedded" if text else "none"
    language_distribution = detect_distribution(str(text) if text else None)
    language = _language(language_distribution, result)
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO lyrics (track_id, source, text, language, provenance, availability_status)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (track_id) DO UPDATE SET source=EXCLUDED.source, text=EXCLUDED.text,
                 language=EXCLUDED.language, provenance=EXCLUDED.provenance,
                 availability_status=EXCLUDED.availability_status, created_at=now()
               WHERE (EXCLUDED.source != 'transcribed' AND EXCLUDED.text IS NOT NULL)
                  OR NULLIF(btrim(lyrics.text), '') IS NULL
               RETURNING id""",
            (
                track_id,
                source,
                text,
                language,
                Jsonb(_provenance(result, language_distribution)),
                status,
            ),
        )
        row = cursor.fetchone()
        return row[0] if row else None


def _store_embeddings(
    connection: psycopg.Connection, track_id: uuid.UUID, run_id: uuid.UUID, result
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM embeddings WHERE track_id=%s AND run_id=%s AND embedding_type='lyrics'",
            (track_id, run_id),
        )
        cursor.execute(
            """INSERT INTO embeddings
                 (track_id, run_id, embedding_type, dimension, aggregation, embedding, inference_ms, peak_vram_bytes)
               VALUES (%s,%s,'lyrics',%s,'normalized_mean',%s::vector,%s,%s)""",
            (
                track_id,
                run_id,
                len(result.aggregate),
                _vector_literal(result.aggregate),
                result.inference_ms,
                result.peak_vram_bytes,
            ),
        )
        for index, (vector, token_range) in enumerate(zip(result.windows, result.token_ranges)):
            cursor.execute(
                """INSERT INTO embeddings
                     (track_id, run_id, embedding_type, window_index, dimension, aggregation, embedding)
                   VALUES (%s,%s,'lyrics',%s,%s,%s,%s::vector)""",
                (
                    track_id,
                    run_id,
                    index,
                    len(vector),
                    f"tokens:{token_range[0]}-{token_range[1]}",
                    _vector_literal(vector),
                ),
            )


class LyricsItem(Item):
    """A song whose lyrics or lyric embedding are missing or stale."""

    track_id: uuid.UUID
    external_id: str


@dataclass(frozen=True)
class LyricsResult:
    """A song's lyrics, and whether they still need storing (``False``: already stored)."""

    lyrics: dict[str, object]
    store: bool = True


@dataclass(frozen=True)
class _Transcription:
    item: LyricsItem
    config: tuple
    missing: dict[str, object]
    """The song's lyrics if it is not transcribed after all (status missing)."""


class LyricsProcessor(Processor[None, LyricsItem, LyricsResult]):
    """Retrieves the batch's lyrics from Navidrome, and transcribes the songs that have none.

    Phases (docs/batch-phases.md): retrieve every song's lyrics; separate the vocals of every song
    to transcribe with one Roformer; generate every transcript with MOSS loaded once; repair
    compressed timing with the karaoke aligner loaded once, only for drafts that need it. Each
    song's lyrics are stored as soon as they are known. Each song counts as its lyrics status
    (available, missing, unavailable, instrumental, ...), except lyrics a concurrent writer
    superseded, which count as nothing and are not embedded.

    Afterwards ``embeddable`` holds the songs whose text needs embedding, in order, and
    ``fresh`` the track ids transcribed in this batch.
    """

    name = "lyrics"
    summary_key = "lyrics"
    counters = ("available", "missing", "unavailable")
    activity = "Retrieving lyrics for"
    current_message = "Lyrics analysis already current"

    def __init__(self) -> None:
        super().__init__()
        self.embeddable: list[tuple[uuid.UUID, str, dict[str, object]]] = []
        self.fresh: set[uuid.UUID] = set()

    def plan(self, context: ProcessingContext) -> list[LyricsItem]:
        ids = list(context.track_ids)
        with connect() as connection:
            self.library_id = resolve_library_id(connection, context.library_url)
            configure_representations(connection)
            planned = plan_lyrics(connection, ids, library_id=self.library_id).lyrics_external_ids
            if context.options.get("refresh_lyrics", False):
                planned = frozenset(ids)
            if not planned:
                return []
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT DISTINCT ON (ts.track_id) ts.track_id, ts.external_id, t.title
                       FROM track_sources ts JOIN tracks t ON t.id=ts.track_id
                       WHERE ts.source_type='subsonic' AND ts.external_id=ANY(%s) AND ts.library_id=%s
                       ORDER BY ts.track_id, ts.id""",
                    (list(planned), self.library_id),
                )
                rows = cursor.fetchall()
        return [
            LyricsItem(
                key=str(track_id), label=title or "", track_id=track_id, external_id=external_id
            )
            for track_id, external_id, title in rows
        ]

    def open(self, context, items) -> None:
        self.stack = ExitStack()
        self.connection = self.stack.enter_context(connect())
        self.client = self.stack.enter_context(NavidromeClient(*context.library))

    def close(self) -> None:
        if hasattr(self, "stack"):
            self.stack.close()

    def process(self, context, items):
        transcriptions: list[_Transcription] = []
        for index, item in enumerate(items):
            context.check()
            try:
                found = self._retrieve(item)
            except Exception as error:  # noqa: BLE001 - one song's lyrics
                self.connection.rollback()
                logger.exception("Lyrics retrieval/transcription failed for %s", item.track_id)
                yield ItemResult.failed(item, _failure("Lyrics retrieval failed", error))
                continue
            if isinstance(found, _Transcription):
                transcriptions.append(found)
                context.publish(
                    {
                        "phase": "lyrics",
                        "message": f"Lyrics need transcription for {item.label}",
                        "completed": index + 1,
                        "total": len(items),
                        "unit": "tracks",
                    }
                )
                continue
            yield ItemResult(item=item, outcome=None, output=found)
        if transcriptions:
            yield from self._transcribe(context, transcriptions)

    def _retrieve(self, item: LyricsItem) -> LyricsResult | _Transcription:
        """The song's lyrics, or the transcription it needs."""
        from worker.transcription.config import transcription_enabled, transcription_model

        connection = self.connection
        provider_lyrics = self.client.lyrics(item.external_id)
        # Archive the provider's lyrics as soon as they are read, whatever the song's outcome.
        store_original(
            connection, self.library_id, item.external_id, item.track_id, provider_lyrics
        )
        connection.commit()
        # Manual edits remain authoritative when their embedding needs to be rebuilt.
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT text, provenance, availability_status, source FROM lyrics WHERE track_id=%s",
                (item.track_id,),
            )
            stored = cursor.fetchone()
        if stored and (stored[1] or {}).get("manual_status"):
            # A user-set status is final, even though these rows use source='none'.
            lyrics = {
                "text": stored[0],
                "status": (stored[1] or {})["manual_status"],
                **(stored[1] or {}),
            }
            return LyricsResult(lyrics, store=False)
        if stored and stored[3] == "manual":
            return LyricsResult(
                {"text": stored[0], "status": stored[2], **(stored[1] or {})}, store=False
            )
        if stored and (stored[1] or {}).get("forced_transcription"):
            forced_language = str((stored[1] or {}).get("transcription_language") or "")
            config = transcription_model() if transcription_enabled(connection) else None
            if not config:
                raise RuntimeError("AI lyric generation is disabled")
            connection.commit()
            missing = {"status": "missing", "transcription_language": forced_language}
            return _Transcription(item, config, missing)
        lyrics = provider_lyrics
        if str(lyrics.get("text") or "").strip():
            return LyricsResult(lyrics)
        # A retrieval miss must not erase a previous AI transcript or lyrics that merely need
        # embedding with a newer embedding model.
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT text, provenance, availability_status FROM lyrics WHERE track_id=%s AND (NULLIF(btrim(text),'') IS NOT NULL OR availability_status='instrumental')",
                (item.track_id,),
            )
            stored = cursor.fetchone()
        if stored:
            return LyricsResult(
                {"text": stored[0], "status": stored[2], **(stored[1] or {})}, store=False
            )
        if lyrics.get("status") == "instrumental":
            return LyricsResult(lyrics)
        if evidence := _instrumental_evidence(connection, item.track_id):
            # Clearly no sung vocals: transcribing the near-silent stem only yields invented
            # text. Stored as instrumental, so later syncs skip it.
            return LyricsResult(
                {"text": None, "status": "instrumental", "instrumental_evidence": evidence}
            )
        config = transcription_model() if transcription_enabled(connection) else None
        if config:
            connection.commit()
            return _Transcription(item, config, lyrics)
        return LyricsResult(lyrics)

    def _transcribe(self, context, transcriptions: list[_Transcription]):
        """Separate, generate and repair every transcription, each phase with one model."""
        from worker.transcription.config import transcription_enabled

        connection, client, remote = self.connection, self.client, context.remote
        ready: list[_Transcription] = []
        digests: dict[uuid.UUID, str] = {}
        # Only genuine fallbacks reach this phase. Never separate songs solely because their
        # supplied lyrics need a new embedding.
        with separation_phase():
            for index, candidate in enumerate(transcriptions):
                item = candidate.item
                update = {"phase": "preprocess", "total": len(transcriptions), "unit": "tracks"}
                context.publish(
                    {**update, "message": f"Preparing vocals for {item.label}", "completed": index}
                )
                try:
                    if not transcription_enabled(connection):
                        yield ItemResult(
                            item=item, outcome=None, output=LyricsResult(candidate.missing)
                        )
                        continue
                    audio = client.audio_bytes(item.external_id)
                    if remote is None:
                        prepare_audio(audio, vocals=True, check=context.check)
                    else:
                        # Vocals are separated on Modal; only the source audio goes up.
                        digests[item.track_id] = remote.upload_audio(audio)
                    ready.append(candidate)
                    connection.commit()
                except Exception as error:  # noqa: BLE001 - one song's audio or separation
                    connection.rollback()
                    logger.exception("Vocal preparation failed for %s", item.track_id)
                    yield ItemResult.failed(item, _failure("Vocal preparation failed", error))
                context.publish({**update, "completed": index + 1})
        if remote is not None and ready:
            separated = remote.prepare_vocals(
                [digests[candidate.item.track_id] for candidate in ready], vocals=True
            )
            kept = []
            for candidate, outcome in zip(list(ready), separated):
                if isinstance(outcome, BaseException):
                    logger.error("Vocal separation on Modal failed for %s", candidate.item.track_id)
                    yield ItemResult.failed(
                        candidate.item, _failure("Vocal separation failed", outcome)
                    )
                else:
                    kept.append(candidate)
            ready = kept

        transcribers: dict[tuple, SongTranscriber] = {}
        repairs = []
        try:
            for index, candidate in enumerate(ready):
                item, title = candidate.item, candidate.item.label
                try:
                    if not transcription_enabled(connection):
                        yield ItemResult(
                            item=item, outcome=None, output=LyricsResult(candidate.missing)
                        )
                        continue
                    transcriber = transcribers.get(tuple(candidate.config))
                    if transcriber is None:
                        transcriber = transcribers[tuple(candidate.config)] = SongTranscriber(
                            *candidate.config
                        )
                        if remote is None:
                            # MOSS loads once for every song of this phase.
                            transcriber.open()
                    context.publish(
                        {
                            "phase": "transcription",
                            "message": f"Transcribing lyrics for {title}",
                            "completed": index,
                            "total": len(ready),
                            "unit": "tracks",
                        }
                    )
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """SELECT a.activity FROM track_vocal_activity a
                            JOIN current_embeddings e ON e.track_id=a.track_id AND e.run_id=a.run_id
                            WHERE a.track_id=%s AND e.embedding_type='voice-gender'
                            ORDER BY a.created_at DESC LIMIT 1""",
                            (item.track_id,),
                        )
                        activity_row = cursor.fetchone()
                    arguments = {
                        "language": str(candidate.missing.get("transcription_language") or "")
                        or None,
                        "check": lambda: context.publish({"phase": "transcription"}),
                        "progress": lambda detail, title=title: context.publish(
                            {"phase": "transcription", "message": f"{title}: {detail}"}
                        ),
                        "diagnostic_sink": diagnostic_writer(item.track_id),
                        "vocal_activity": activity_row[0] if activity_row else None,
                    }
                    audio = client.audio_bytes(item.external_id)
                    if remote is None:
                        draft = transcriber.generate(audio, **arguments)
                    else:
                        draft = remote.transcribe_generate(audio, **arguments)
                    if needs_repair(draft):
                        # Stored after the timing repair phase.
                        repairs.append((item, transcriber, draft))
                        continue
                    lyrics = transcriber.finish(draft)
                except Exception as error:  # noqa: BLE001 - one song's transcription
                    connection.rollback()
                    logger.exception("Lyrics transcription failed for %s", item.track_id)
                    yield ItemResult.failed(item, _failure("Transcription failed", error))
                    continue
                yield ItemResult(item=item, outcome=None, output=LyricsResult(lyrics))
        finally:
            for transcriber in transcribers.values():
                transcriber.close()

        if not repairs:
            return
        # Timing repair phase: the karaoke aligner, loaded once, for drafts that need it.
        finished = (
            iter(remote.transcribe_finish([draft for _, _, draft in repairs]))
            if remote is not None
            else None
        )
        try:
            for index, (item, transcriber, draft) in enumerate(repairs):
                context.publish(
                    {
                        "phase": "transcription",
                        "message": f"Repairing timing for {item.label}",
                        "completed": index,
                        "total": len(repairs),
                        "unit": "tracks",
                    }
                )
                try:
                    lyrics = transcriber.finish(draft) if finished is None else next(finished)
                    if isinstance(lyrics, BaseException):
                        raise lyrics
                except Exception as error:  # noqa: BLE001 - one song's timing repair
                    connection.rollback()
                    logger.exception("Lyrics timing repair failed for %s", item.track_id)
                    yield ItemResult.failed(item, _failure("Timing repair failed", error))
                    continue
                yield ItemResult(item=item, outcome=None, output=LyricsResult(lyrics))
        finally:
            if finished is None:
                from worker.kara.pipeline import _stop_fa_kara_worker

                _stop_fa_kara_worker()

    def persist(self, context, item: LyricsItem, output: LyricsResult) -> tuple[str, ...]:
        lyrics = output.lyrics
        try:
            if output.store and _store_lyrics(self.connection, item.track_id, lyrics) is None:
                # A concurrent writer supplied text. Never embed our discarded candidate.
                self.connection.commit()
                return ()
            if lyrics.get("text"):
                self.embeddable.append((item.track_id, item.label, lyrics))
                if output.store and lyrics.get("ai_generated") and lyrics.get("lines"):
                    self.fresh.add(item.track_id)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return (str(lyrics.get("status") or "unavailable"),)


class RefinementItem(Item):
    """A transcript made in this batch."""

    track_id: uuid.UUID
    lyrics: dict[str, object]


class TranscriptRefinementProcessor(Processor[None, RefinementItem, dict[str, object]]):
    """External AI second pass over a batch's new AI transcripts.

    Runs after transcription has released its models and before embedding, so search,
    translations and karaoke all use the corrected text. A failure keeps the transcript
    (``refinement_failed``); a transcript changed meanwhile is left alone. ``refined`` maps the
    refined songs to their new lyrics.
    """

    name = "lyrics"
    summary_key = "lyrics"
    total_key = None
    activity = "Refining transcript for"
    current_message = None

    def __init__(self, transcripts: list[RefinementItem]) -> None:
        super().__init__()
        self.transcripts = transcripts
        self.refined: dict[uuid.UUID, dict[str, object]] = {}

    def plan(self, context: ProcessingContext) -> list[RefinementItem]:
        from core.lyrics.translations import load_settings
        from worker.transcription.refinement import enabled

        try:
            settings, encrypted = load_settings()
        except Exception:
            logger.exception("Could not load External AI settings; transcripts stay unrefined")
            return []
        if not enabled(settings):
            return []
        self.settings = settings
        self.key = credential_cipher().decrypt(bytes(encrypted)).decode() if encrypted else None
        return list(self.transcripts)

    def open(self, context, items) -> None:
        self.stack = ExitStack()
        self.connection = self.stack.enter_context(connect())

    def close(self) -> None:
        if hasattr(self, "stack"):
            self.stack.close()

    def process(self, context, items):
        from worker.transcription.refinement import RefinementError, refine

        for index, item in enumerate(items):
            context.check()
            context.publish(
                {
                    "phase": "lyrics",
                    "message": f"Refining transcript for {item.label}",
                    "completed": index,
                    "total": len(items),
                    "unit": "tracks",
                }
            )
            artist = self.connection.execute(
                "SELECT artist FROM tracks WHERE id=%s", (item.track_id,)
            ).fetchone()
            try:
                refined = refine(
                    self.settings,
                    self.key,
                    item.lyrics,
                    title=item.label,
                    artist=(artist[0] if artist else "") or "",
                    languages=(
                        detect_distribution(str(item.lyrics["text"])).get("languages") or {}
                    ),
                )
            except RefinementError as error:
                logger.warning("Transcript refinement failed for %s: %s", item.track_id, error)
                yield ItemResult(item=item, outcome="refinement_failed")
                continue
            yield ItemResult(item=item, outcome=None, output=refined)

    def persist(self, context, item: RefinementItem, output: dict[str, object]) -> tuple[str, ...]:
        distribution = detect_distribution(str(output["text"]))
        try:
            updated = self.connection.execute(
                """UPDATE lyrics SET text=%s, language=%s, provenance=%s
                   WHERE track_id=%s AND source='transcribed' AND text=%s""",
                (
                    output["text"],
                    _language(distribution, output),
                    Jsonb(_provenance(output, distribution)),
                    item.track_id,
                    item.lyrics["text"],
                ),
            ).rowcount
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        if not updated:
            return ()
        self.refined[item.track_id] = output
        return ("refined",)


class EmbeddingItem(Item):
    """A song's lyrics to embed."""

    track_id: uuid.UUID
    text: str


class LyricsEmbeddingProcessor(Processor[None, EmbeddingItem, object]):
    """Embeds lyrics with BGE-M3, loaded once (on Modal, every text in one call).

    A failed song is recorded in the run's attempt, with no lyrics or error text.
    """

    name = "lyrics"
    summary_key = "lyrics"
    counters = ("embedded",)
    activity = "Embedding lyrics for"
    current_message = None

    def __init__(self, songs: list[EmbeddingItem]) -> None:
        super().__init__()
        self.songs = songs

    def plan(self, context: ProcessingContext) -> list[EmbeddingItem]:
        return list(self.songs)

    def open(self, context, items) -> None:
        context.publish(
            {
                "phase": "models",
                "message": "Loading BGE-M3 lyrics model",
                "completed": 0,
                "total": 1,
                "unit": "models",
            }
        )
        config = EmbeddingConfig.load()
        if context.remote is not None:
            self.model = RemoteModel(
                LyricsEmbeddingModel.name, config.lyrics_model_id, config.lyrics_revision
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = LyricsEmbeddingModel(
                config.lyrics_model_id, config.lyrics_revision, device
            )
        self.stack = ExitStack()
        self.connection = self.stack.enter_context(connect())
        self.run_id = _create_run(self.connection, self.model)
        self.attempt_id = start_attempt(self.connection, self.run_id, len(items))
        self.connection.commit()

    def process(self, context, items):
        # On Modal, every text is embedded in one call; results stream back in order.
        remote_outcomes = (
            iter(context.remote.embed_lyrics([item.text for item in items]))
            if context.remote is not None
            else None
        )
        for item in items:
            context.check()
            try:
                if remote_outcomes is None:
                    embedded = self.model.embed(item.text)
                else:
                    embedded = next(remote_outcomes)
                    if isinstance(embedded, BaseException):
                        raise embedded
            except Exception as error:  # noqa: BLE001 - one song's embedding
                self._record_failure(item)
                yield ItemResult.failed(item, _failure("Lyrics embedding failed", error))
                continue
            yield ItemResult(item=item, outcome="embedded", output=embedded)
        finish_attempt(self.connection, self.attempt_id)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE analysis_runs SET status='complete', finished_at=now() WHERE id=%s",
                (self.run_id,),
            )
        self.connection.commit()

    def _record_failure(self, item: EmbeddingItem) -> None:
        self.connection.rollback()
        record_track(
            self.connection,
            self.attempt_id,
            str(item.track_id),
            item.track_id,
            error="Lyrics embedding failed; see service logs",
        )
        self.connection.commit()

    def persist(self, context, item: EmbeddingItem, output) -> None:
        try:
            _store_embeddings(self.connection, item.track_id, self.run_id, output)
            record_track(self.connection, self.attempt_id, str(item.track_id), item.track_id)
            self.connection.commit()
        except Exception:
            self._record_failure(item)
            raise

    def close(self) -> None:
        if hasattr(self, "stack"):
            self.stack.close()
        if getattr(self, "model", None) is not None:
            release_model(self.model)
            self.model = None


def lyrics_stage(context: ProcessingContext) -> dict[str, object]:
    """The lyrics stage: retrieve and transcribe, refine new transcripts, embed.

    Returns the stage's summary with the keys it always had: total, available, missing,
    unavailable, embedded and failed, plus other statuses and refinement counts once counted.
    """
    retrieval = LyricsProcessor()
    found = run_stage(retrieval, context).as_dict()
    summary: dict[str, object] = {
        key: found[key] for key in ("total", "available", "missing", "unavailable")
    }
    summary["embedded"] = 0
    failed = int(found["failed"])
    extra = {key: value for key, value in found.items() if key not in summary and key != "failed"}
    embeddable = retrieval.embeddable
    fresh = [
        RefinementItem(key=str(track_id), label=title, track_id=track_id, lyrics=lyrics)
        for track_id, title, lyrics in embeddable
        if track_id in retrieval.fresh
    ]
    if fresh:
        refinement = TranscriptRefinementProcessor(fresh)
        refined = run_stage(refinement, context).as_dict()
        failed += int(refined.pop("failed"))
        extra.update(refined)
        embeddable = [
            (track_id, title, refinement.refined.get(track_id, lyrics))
            for track_id, title, lyrics in embeddable
        ]
    if embeddable:
        songs = [
            EmbeddingItem(
                key=str(track_id), label=title, track_id=track_id, text=str(lyrics["text"])
            )
            for track_id, title, lyrics in embeddable
        ]
        embedded = run_stage(LyricsEmbeddingProcessor(songs), context).as_dict()
        summary["embedded"] = embedded["embedded"]
        failed += int(embedded["failed"])
    summary["failed"] = failed
    summary.update(extra)
    return summary


def _failure(message: str, error: BaseException) -> ProcessingError:
    return ProcessingError(message, stage=type(error).__name__)
