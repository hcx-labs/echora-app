"""Index a user's songs into a hum corpus (``hum_corpus``): melody contours per song."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import psycopg

from core.db.connection import connect
from core.errors import InputUnavailableError, ProcessingError
from core.integrations.navidrome import NavidromeClient
from worker.library.sources import source_rows
from worker.processing import Item, ItemResult, ProcessingContext, Processor

Contours = dict[str, tuple[np.ndarray, np.ndarray]]


class CorpusSong(Item):
    """A song of the corpus's batch."""

    track_id: uuid.UUID
    external_id: str


@dataclass(frozen=True)
class CorpusEntry:
    """Contours to store for a song, or ``None`` when the run already has them."""

    contours: Contours | None


class MelodyCorpusProcessor(Processor[None, CorpusSong, CorpusEntry]):
    """Extracts the batch's melody contours for the corpus's run and adds the songs to the corpus.

    ``tracks`` counts songs added, ``contours`` the contours stored (one per audio source).
    """

    name = "melody-index"
    summary_key = "melody"
    counters = ("tracks", "contours")
    total_key = None
    activity = "Extracting melody contours"

    def plan(self, context: ProcessingContext) -> list[CorpusSong]:
        self.corpus_id = context.options["corpus_id"]
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM hum_corpora WHERE id=%s AND user_id=%s",
                (self.corpus_id, context.user_id),
            )
            if not cursor.fetchone():
                raise ValueError("Unknown hum corpus")
        rows = source_rows(context.user_id, context.library_url, list(context.track_ids))
        return [
            CorpusSong(key=str(track_id), track_id=track_id, external_id=external_id)
            for external_id, track_id in rows
        ]

    def open(self, context: ProcessingContext, items: Sequence[CorpusSong]) -> None:
        from worker.melody.index import create_sync_run

        self.connection: psycopg.Connection = connect()
        self.run_id = create_sync_run(self.connection)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE hum_corpora SET run_id=%s WHERE id=%s AND user_id=%s",
                (self.run_id, self.corpus_id, context.user_id),
            )
        self.connection.commit()
        self.client = NavidromeClient(*context.library)
        self.client.__enter__()

    def process(
        self, context: ProcessingContext, items: Sequence[CorpusSong]
    ) -> Iterator[ItemResult[CorpusSong, CorpusEntry]]:
        for item in items:
            context.check()
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM melody_contours WHERE track_id=%s AND run_id=%s",
                    (item.track_id, self.run_id),
                )
                current = bool(cursor.fetchone()[0])
            if current:
                yield ItemResult(item=item, outcome="tracks", output=CorpusEntry(None))
                continue
            try:
                audio = self.client.audio_bytes(item.external_id)
            except Exception:  # noqa: BLE001 - one song's audio
                yield ItemResult.failed(item, InputUnavailableError("Audio unavailable"))
                continue
            try:
                contours = self._contours(context, audio)
            except Exception as error:  # noqa: BLE001 - one song's separation or extraction
                failure = ProcessingError("Melody extraction failed", stage=type(error).__name__)
                yield ItemResult.failed(item, failure)
                continue
            yield ItemResult(item=item, outcome="tracks", output=CorpusEntry(contours))

    @staticmethod
    def _contours(context: ProcessingContext, audio: bytes) -> Contours:
        from worker.media.preprocessing import prepare_audio
        from worker.melody.index import track_contours

        context.publish({"phase": "preprocess", "message": "Preparing shared melody audio"})
        prepare_audio(audio, mono_rates=(44100,), stereo=True, melody=True, check=context.check)
        context.publish({"phase": "melody-index", "message": "Extracting melody contours"})
        if context.remote is not None:
            return context.remote.melody_one(audio)
        return track_contours(audio)

    def persist(
        self, context: ProcessingContext, item: CorpusSong, output: CorpusEntry
    ) -> list[str]:
        from worker.melody.index import store_track_contours

        try:
            stored = 0
            if output.contours is not None:
                stored = store_track_contours(
                    self.connection, item.track_id, self.run_id, b"", output.contours
                )
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO hum_corpus_tracks(corpus_id,track_id) VALUES (%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (self.corpus_id, item.track_id),
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return ["contours"] * stored

    def close(self) -> None:
        if hasattr(self, "client"):
            self.client.__exit__(None, None, None)
        if hasattr(self, "connection"):
            self.connection.close()
