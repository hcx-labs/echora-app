"""Fingerprint songs and group recordings of the same performance (``recordings_backfill``)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import psycopg

from core.db.connection import connect
from core.errors import InputUnavailableError, ProcessingError
from core.integrations.navidrome import NavidromeClient
from core.library import store
from worker.library.fingerprints import compute_chromaprint, store_fingerprint
from worker.library.sources import source_rows
from worker.processing import Item, ItemResult, ProcessingContext, Processor


class FingerprintItem(Item):
    """A song without a stored fingerprint."""

    track_id: uuid.UUID
    external_id: str
    duration: float
    """The library's duration, used when fpcalc measures none."""


@dataclass(frozen=True)
class Fingerprint:
    """A computed chromaprint and the duration it was measured over."""

    fingerprint: bytes
    duration: float
    version: str


class FingerprintProcessor(Processor[None, FingerprintItem, Fingerprint]):
    """Fingerprints the batch's songs that have none and groups each with a recording it matches.

    ``matched`` counts fingerprinted songs that joined a recording group.
    """

    name = "fingerprint"
    summary_key = "recordings"
    counters = ("fingerprinted", "matched")
    total_key = None
    activity = "Fingerprinting"

    def plan(self, context: ProcessingContext) -> list[FingerprintItem]:
        rows = source_rows(context.user_id, context.library_url, list(context.track_ids))
        items = []
        with connect() as connection:
            for external_id, track_id in rows:
                context.check()
                row = store.unfingerprinted_duration(connection, track_id)
                if row:
                    items.append(
                        FingerprintItem(
                            key=str(track_id),
                            track_id=track_id,
                            external_id=external_id,
                            duration=float(row["duration_seconds"] or 0),
                        )
                    )
        return items

    def open(self, context: ProcessingContext, items: Sequence[FingerprintItem]) -> None:
        self.client = NavidromeClient(*context.library)
        self.client.__enter__()
        self.connection: psycopg.Connection = connect()

    def process(
        self, context: ProcessingContext, items: Sequence[FingerprintItem]
    ) -> Iterator[ItemResult[FingerprintItem, Fingerprint]]:
        for item in items:
            context.check()
            # A song listed under two source ids is planned twice: the second is the fallback
            # when the first could not be read, and is skipped once the song has a fingerprint.
            if store.unfingerprinted_duration(self.connection, item.track_id) is None:
                yield ItemResult(item=item, outcome=None)
                continue
            try:
                audio = self.client.audio_bytes(item.external_id)
            except Exception:  # noqa: BLE001 - one song's audio
                yield ItemResult.failed(item, InputUnavailableError("Audio unavailable"))
                continue
            try:
                fingerprint, measured, version = compute_chromaprint(audio)
            except Exception as error:  # noqa: BLE001 - fpcalc failed or timed out on this song
                failure = ProcessingError("Fingerprinting failed", stage=type(error).__name__)
                yield ItemResult.failed(item, failure)
                continue
            output = Fingerprint(fingerprint, measured or item.duration, version)
            yield ItemResult(item=item, outcome="fingerprinted", output=output)

    def persist(
        self, context: ProcessingContext, item: FingerprintItem, output: Fingerprint
    ) -> tuple[str, ...]:
        try:
            match = store_fingerprint(
                self.connection, item.track_id, output.fingerprint, output.duration, output.version
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return ("matched",) if match.get("matched") else ()

    def close(self) -> None:
        if hasattr(self, "connection"):
            self.connection.close()
        if hasattr(self, "client"):
            self.client.__exit__(None, None, None)
