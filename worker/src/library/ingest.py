"""Navidrome ingest: resolves songs to tracks, then runs the analysis phases they need."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psycopg
import torch

from core.analysis import store as runs
from core.db.connection import connect
from core.embeddings import store as embeddings
from core.embeddings.audio_models import (
    AudioEmbeddingModel,
    EmbeddingResult,
    MertModel,
    MuQMuLanModel,
    release_model,
)
from core.embeddings.config import EmbeddingConfig
from core.embeddings.representations import configure_representations, embedding_config
from core.errors import InputUnavailableError, ProcessingError
from core.integrations.navidrome import NavidromeClient, NavidromeTrack
from core.library import store as library
from worker.library.attempts import finish_attempt, record_track, start_attempt
from worker.library.descriptors import analyze_audio_descriptors, store_audio_descriptors
from worker.library.fingerprints import compute_chromaprint, store_fingerprint
from worker.library.visual_features import analyze_visual_features, store_visual_features
from worker.library.waveforms import analyze_waveform, store_waveform
from worker.media.audio import decode_audio, full_coverage_window_ranges
from worker.media.preprocessing import active_cache, get_check, prepare_audio
from worker.melody.index import create_sync_run, store_track_contours, track_contours
from worker.planning.audio import audio_prerequisites, plan_audio
from worker.processing import Item, ItemResult, ProcessingContext, Processor, run_stage
from worker.remote.session import RemoteModel, current as current_remote

CONTENT_NAMESPACE = uuid.UUID("0c300f5d-99d1-48b8-b12a-a52b9556be86")
logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    """What an ingest batch did, counted per phase."""

    requested: int = 0
    discovered: int = 0
    downloaded: int = 0
    inserted: int = 0
    already_linked: int = 0
    reused_embeddings: int = 0
    embedded_muq: int = 0
    embedded_mert: int = 0
    fingerprinted: int = 0
    waveforms_generated: int = 0
    visual_features_generated: int = 0
    melody_indexed: int = 0
    melody_contours: int = 0
    recording_matches: int = 0
    recording_fingerprinted: int = 0
    described_audio: int = 0
    failed: int = 0


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _config_hash(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _create_run(connection: psycopg.Connection, model: AudioEmbeddingModel) -> uuid.UUID:
    config = embedding_config(model.name, model.revision)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return embeddings.audio_embedding_run(
        connection,
        model.name,
        model.revision,
        _config_hash(config),
        config,
        environment,
        str(model.device),
    )


def _has_fingerprint(connection: psycopg.Connection, track_id: uuid.UUID) -> bool:
    return library.has_fingerprint(connection, track_id)


def _model_has_embedding(
    connection: psycopg.Connection, track_id: uuid.UUID, run_id: uuid.UUID
) -> bool:
    return embeddings.has_audio_track_embedding(connection, track_id, run_id)


def embed_track(
    audio: bytes,
    model: AudioEmbeddingModel,
) -> tuple[EmbeddingResult, list[tuple[float, float]]]:
    """Decode, window and embed one track. The same code runs locally and on Modal."""
    waveform = decode_audio(audio)
    ranged_windows = full_coverage_window_ranges(waveform)
    del waveform
    windows = [item[0] for item in ranged_windows]
    ranges = [(item[1], item[2]) for item in ranged_windows]
    return model.embed_windows(windows), ranges


def _store_embedding(
    connection: psycopg.Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    result: EmbeddingResult,
    ranges: list[tuple[float, float]] | None = None,
) -> None:
    embeddings.save_audio_embeddings(
        connection,
        track_id,
        run_id,
        {
            "dimension": len(result.vector),
            "embedding": _vector_literal(result.vector),
            "inference_ms": result.inference_ms,
            "peak_vram_bytes": result.peak_vram_bytes,
        },
        [
            (
                window_index,
                ranges[window_index][0] if ranges else None,
                ranges[window_index][1] if ranges else None,
                len(vector),
                _vector_literal(vector),
            )
            for window_index, vector in enumerate(result.window_vectors)
        ],
    )


def _source_track_id(
    connection: psycopg.Connection, library_id: uuid.UUID, external_id: str
) -> uuid.UUID | None:
    return library.source_track_id(connection, library_id, external_id)


def _upsert_track(
    connection: psycopg.Connection, library_id: uuid.UUID, song: NavidromeTrack, audio_hash: str
) -> tuple[uuid.UUID, bool]:
    from core.library.visibility import lock_library, source_remapped

    lock_library(connection, library_id)
    previous_id = _source_track_id(connection, library_id, song.id)
    track_id = uuid.uuid5(CONTENT_NAMESPACE, audio_hash)
    metadata = {"navidrome": song.raw, "identity": "sha256-source-bytes"}
    inserted = library.add_track(
        connection,
        track_id,
        audio_hash,
        title=song.title,
        artist=song.artist,
        album=song.album,
        year=song.year,
        duration_seconds=song.duration,
        genres=[song.genre] if song.genre else [],
        metadata=metadata,
    )
    canonical_id = library.track_with_hash(connection, audio_hash)
    library.bind_source(connection, library_id, canonical_id, song.path, song.id, song.raw)
    if previous_id != canonical_id:
        source_remapped(connection, library_id, song.id)
    return canonical_id, inserted


def _library(connection: psycopg.Connection, url: str) -> uuid.UUID:
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/"))
    return library.library_id(connection, url.rstrip("/"), namespace)


def _bound_source_audio(
    client: NavidromeClient, external_id: str, binding: tuple[uuid.UUID, str]
) -> tuple[bytes, uuid.UUID]:
    track_id, expected_hash = binding
    audio = client.audio_bytes(external_id)
    if hashlib.sha256(audio).hexdigest() != expected_hash:
        raise ValueError("Source content changed after identity resolution; retry this source")
    return audio, track_id


@dataclass
class IngestBatch:
    """What every ingest phase shares: the connection, the library and the resolved songs.

    ``bindings`` ties each song to the track and the exact bytes its identity was resolved
    from; later phases read the song again and reject other bytes.
    """

    connection: psycopg.Connection
    navidrome: NavidromeClient
    library_id: uuid.UUID
    verify_audio_hashes: bool = True
    songs: list[NavidromeTrack] = field(default_factory=list)
    bindings: dict[str, tuple[uuid.UUID, str]] = field(default_factory=dict)

    def audio(self, song: NavidromeTrack) -> tuple[bytes, uuid.UUID]:
        """The song's bytes, checked against its identity, and its track id."""
        return _bound_source_audio(self.navidrome, song.id, self.bindings[song.id])

    def commit(self, write: Callable[[], Any]) -> Any:
        """Run one song's writes as one transaction."""
        try:
            result = write()
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return result


class SongItem(Item):
    """One Navidrome song of the batch."""

    song: Any
    """The ``NavidromeTrack``."""

    def track(self) -> dict[str, Any]:
        """The song as the setup screen names it."""
        return {"id": self.song.id, "title": self.song.title, "artist": self.song.artist}


def _song_item(song: NavidromeTrack) -> SongItem:
    return SongItem(key=song.id, label=song.title or "", song=song)


def _failure(message: str, error: BaseException) -> ProcessingError:
    return ProcessingError(message, stage=type(error).__name__)


class _SongPhase(Processor[None, SongItem, Any]):
    """One phase of ingest over the resolved songs it selects (by external id)."""

    total_key = None
    current_message = None
    announces_items = True
    failure_message = "Analysis failed"

    def __init__(self, batch: IngestBatch, external_ids: frozenset[str] = frozenset()) -> None:
        super().__init__()
        self.batch = batch
        self.external_ids = external_ids

    def plan(self, context: ProcessingContext) -> list[SongItem]:
        return [_song_item(song) for song in self.batch.songs if song.id in self.external_ids]

    def process(self, context, items):
        for item in items:
            context.check()
            try:
                audio, track_id = self.batch.audio(item.song)
                result = self.analyze(context, audio)
                del audio
            except Exception as error:  # noqa: BLE001 - one song's audio or analysis
                self.log_failure(item)
                yield ItemResult.failed(item, _failure(self.failure_message, error))
                continue
            yield ItemResult(item=item, outcome=self.outcome(result), output=(track_id, result))

    def analyze(self, context: ProcessingContext, audio: bytes) -> Any:
        raise NotImplementedError

    def outcome(self, result: Any) -> str:
        return self.counters[0]

    def log_failure(self, item: SongItem) -> None:
        logger.exception("%s failed for Navidrome song %s", self.failure_message, item.key)


class IdentityPhase(_SongPhase):
    """Resolves each song to a canonical track and binds later phases to its bytes.

    The track comes from the song's bytes or, without hash checks, from its known source.
    """

    name = "identity"
    summary_key = "identity"
    counters = ("downloaded", "inserted")
    activity = "Resolving track identity"

    def __init__(self, batch: IngestBatch, songs: list[NavidromeTrack]) -> None:
        super().__init__(batch)
        self.found = songs

    def plan(self, context):
        return [_song_item(song) for song in self.found]

    def process(self, context, items):
        connection = self.batch.connection
        for item in items:
            context.check()
            song = item.song
            try:
                if not self.batch.verify_audio_hashes:
                    known = library.known_source(connection, self.batch.library_id, song.id)
                    if known:
                        # Reuse identity, but do not claim the bytes were verified.
                        output = ("known", known["track_id"], known["audio_hash"])
                        yield ItemResult(item=item, outcome=None, output=output)
                        continue
                audio = self.batch.navidrome.audio_bytes(song.id)
                audio_hash = hashlib.sha256(audio).hexdigest()
                del audio
            except Exception:  # noqa: BLE001 - one unavailable source must not stop the batch
                connection.rollback()
                logger.warning("Could not resolve source identity for Navidrome song %s", song.id)
                yield ItemResult.failed(item, InputUnavailableError("Source unavailable"))
                continue
            yield ItemResult(item=item, outcome="downloaded", output=("hashed", audio_hash))

    def persist(self, context, item, output) -> tuple[str, ...]:
        song, batch = item.song, self.batch
        if output[0] == "known":
            _, track_id, audio_hash = output
            batch.commit(lambda: _refresh_known_source(batch, song, track_id))
            inserted = False
        else:
            audio_hash = output[1]
            track_id, inserted = batch.commit(lambda: _bind_source(batch, song, audio_hash))
        batch.bindings[song.id] = (track_id, audio_hash)
        batch.songs.append(song)
        return ("inserted",) if inserted else ()

    def log_failure(self, item):
        logger.warning("Could not resolve source identity for Navidrome song %s", item.key)


def _refresh_known_source(batch: IngestBatch, song: NavidromeTrack, track_id) -> None:
    library.refresh_source(
        batch.connection,
        batch.library_id,
        song.id,
        song.raw,
        track_id,
        title=song.title,
        artist=song.artist,
        album=song.album,
        year=song.year,
        duration_seconds=song.duration,
        genres=[song.genre] if song.genre else [],
    )


def _bind_source(batch: IngestBatch, song: NavidromeTrack, audio_hash: str):
    track_id, inserted = _upsert_track(batch.connection, batch.library_id, song, audio_hash)
    library.mark_source_verified(batch.connection, batch.library_id, song.id, track_id)
    return track_id, inserted


def _prepare_audio(context: ProcessingContext, batch: IngestBatch, plan) -> None:
    """Prepare only the formats outstanding tasks need, before any GPU model loads.

    Same-format consumers and later batches read the disk cache. A song that cannot be
    prepared is left to its consumers, which count and retry it.
    """
    if active_cache() is None:
        return
    preparing = [(song, audio_prerequisites(plan, song.id)) for song in batch.songs]
    preparing = [
        (song, needs) for song, needs in preparing if needs.mono_rates or needs.stereo_rates
    ]
    from worker.media.roformer import separation_phase

    # Separation phase for melody stems: Roformer loads once, and only if a song needs it.
    with separation_phase():
        for index, (song, needs) in enumerate(preparing):
            update = {"phase": "preprocess", "total": len(preparing), "unit": "tracks"}
            context.publish(
                {**update, "message": f"Preparing audio for {song.title}", "completed": index}
            )
            try:
                audio, _ = batch.audio(song)
                # Melody needs Roformer vocals; on Modal those are separated remotely.
                prepare_audio(
                    audio,
                    mono_rates=needs.mono_rates,
                    stereo_rates=needs.stereo_rates,
                    melody=needs.melody and context.remote is None,
                    check=context.check,
                )
                del audio
                batch.connection.commit()
            except Exception:
                batch.connection.rollback()
                logger.exception("Could not prepare audio for Navidrome song %s", song.id)
            context.publish(
                {**update, "message": f"Prepared audio for {song.title}", "completed": index + 1}
            )


class WaveformPhase(_SongPhase):
    """Stores the waveform peaks the player draws for each selected song."""

    name = "waveform"
    summary_key = "waveform"
    counters = ("waveforms_generated",)
    activity = "Generating waveform for"
    failure_message = "Could not generate waveform"

    def analyze(self, context, audio):
        return analyze_waveform(audio)

    def persist(self, context, item, output):
        self.batch.commit(lambda: store_waveform(self.batch.connection, *output))


class VisualFeaturePhase(_SongPhase):
    """Stores the visualizer's features for each selected song."""

    name = "visual_features"
    summary_key = "visual_features"
    counters = ("visual_features_generated",)
    activity = "Prepared visual features for"
    failure_message = "Could not prepare visual features"

    def analyze(self, context, audio):
        return analyze_visual_features(audio)

    def persist(self, context, item, output):
        self.batch.commit(lambda: store_visual_features(self.batch.connection, *output))


class EmbeddingPhase(_SongPhase):
    """Embeds the selected songs with one audio model, loaded once (on Modal, in one call)."""

    counters = ("reused_embeddings",)
    failure_message = "Audio embedding failed"

    def __init__(
        self,
        batch: IngestBatch,
        external_ids: frozenset[str],
        phase: str,
        label: str,
        load: Callable[[ProcessingContext], AudioEmbeddingModel | RemoteModel],
        models: tuple[str, int, int],
    ) -> None:
        super().__init__(batch, external_ids)
        self.name = self.summary_key = phase
        self.activity = label
        self.counters = (f"embedded_{phase}", "reused_embeddings")
        self.load = load
        self.models = models
        """Progress of model loading: message, models loaded so far, models needed."""

    def open(self, context, items):
        message, loaded, total = self.models
        context.publish(
            {
                "phase": "models",
                "message": message,
                "completed": loaded,
                "total": total,
                "unit": "models",
            }
        )
        self.model = self.load(context)
        connection = self.batch.connection
        self.run_id = _create_run(connection, self.model)
        self.attempt_id = start_attempt(connection, self.run_id, len(items))
        connection.commit()

    def process(self, context, items):
        connection, remote = self.batch.connection, context.remote
        embedded = f"embedded_{self.name}"
        if remote is None:
            for item in items:
                context.check()
                try:
                    audio, track_id = self.batch.audio(item.song)
                    if _model_has_embedding(connection, track_id, self.run_id):
                        yield self._reused(item, track_id)
                        continue
                    outcome = embed_track(audio, self.model)
                    del audio
                except Exception:  # noqa: BLE001 - one song's audio or embedding
                    yield self._failed(item)
                    continue
                yield ItemResult(
                    item=item, outcome=embedded, output=("embedded", track_id, outcome)
                )
        else:
            # Upload each song's audio once, then embed the whole phase in one Modal call.
            # Results stream back in order and are stored as they arrive.
            pending: list[tuple[SongItem, uuid.UUID, str]] = []
            for item in items:
                context.check()
                try:
                    audio, track_id = self.batch.audio(item.song)
                    if _model_has_embedding(connection, track_id, self.run_id):
                        yield self._reused(item, track_id)
                        continue
                    digest = self.batch.bindings[item.song.id][1]
                    remote.upload(digest, audio)
                    del audio
                    pending.append((item, track_id, digest))
                except Exception:  # noqa: BLE001 - one song's audio
                    yield self._failed(item)
            outcomes = remote.embed_audio(self.model, [digest for _, _, digest in pending])
            for (item, track_id, _), outcome in zip(pending, outcomes):
                if isinstance(outcome, BaseException):
                    yield self._failed(item)
                    continue
                yield ItemResult(
                    item=item, outcome=embedded, output=("embedded", track_id, outcome)
                )
        finish_attempt(connection, self.attempt_id)
        runs.complete_run(connection, self.run_id)
        connection.commit()

    @staticmethod
    def _reused(item: SongItem, track_id: uuid.UUID) -> ItemResult:
        return ItemResult(item=item, outcome="reused_embeddings", output=("reused", track_id))

    def _failed(self, item: SongItem) -> ItemResult:
        self.batch.connection.rollback()
        logger.exception("Failed %s phase for Navidrome song %s", self.name, item.key)
        # A failure with an output: persist() records it in the run's attempt.
        return ItemResult(item=item, outcome="failed", output=("error",))

    def persist(self, context, item, output):
        connection, song = self.batch.connection, item.song
        if output[0] == "error":
            self._record_failure(item)
            return
        if output[0] == "reused":
            self.batch.commit(lambda: record_track(connection, self.attempt_id, song.id, output[1]))
            return
        _, track_id, (result, ranges) = output

        def store():
            _store_embedding(connection, track_id, self.run_id, result, ranges)
            record_track(connection, self.attempt_id, song.id, track_id)

        try:
            self.batch.commit(store)
        except Exception:
            self._record_failure(item)
            raise

    def _record_failure(self, item: SongItem) -> None:
        # Do not persist provider exception messages, which may contain credentials.
        record_track(
            self.batch.connection,
            self.attempt_id,
            item.song.id,
            error="Audio embedding failed; see service logs",
        )
        self.batch.connection.commit()

    def close(self):
        model = getattr(self, "model", None)
        if model is not None:
            release_model(model)
            self.model = None


class MelodyPhase(_SongPhase):
    """Extracts melody contours for the hum search.

    On Modal: one separation call and one extraction call for the whole phase.
    """

    name = "melody"
    summary_key = "melody"
    counters = ("melody_indexed",)
    activity = "Extracting melody from"
    failure_message = "Could not extract melody"

    def open(self, context, items):
        self.run_id = create_sync_run(self.batch.connection)
        self.batch.connection.commit()

    def process(self, context, items):
        remote = context.remote
        if remote is None:
            yield from super().process(context, items)
        else:
            unavailable: dict[str, BaseException] = {}
            # Upload once, extract every song's contours in one Modal call; results stream back
            # in song order and are stored as they arrive.
            pending = []
            for item in items:
                try:
                    audio, _ = self.batch.audio(item.song)
                    remote.upload(self.batch.bindings[item.key][1], audio)
                    pending.append(item)
                except Exception as error:  # noqa: BLE001 - one song's audio
                    unavailable[item.key] = error
            # Separation phase on Modal first: one Roformer for every melody song.
            separated = remote.prepare_vocals(
                [self.batch.bindings[item.key][1] for item in pending], melody=True
            )
            for item, outcome in zip(list(pending), separated):
                if isinstance(outcome, BaseException):
                    unavailable[item.key] = outcome
                    pending.remove(item)
            outcomes = iter(remote.melody([self.batch.bindings[item.key][1] for item in pending]))
            for item in items:
                context.check()
                try:
                    outcome = unavailable.get(item.key) or next(outcomes)
                except Exception as error:  # noqa: BLE001 - the Modal stream broke or ended early
                    outcome = error
                if isinstance(outcome, BaseException):
                    self.log_failure(item)
                    yield ItemResult.failed(item, _failure(self.failure_message, outcome))
                    continue
                track_id = self.batch.bindings[item.key][0]
                yield ItemResult(item=item, outcome="melody_indexed", output=(track_id, outcome))
        runs.complete_run(self.batch.connection, self.run_id)
        self.batch.connection.commit()

    def analyze(self, context, audio):
        return track_contours(audio)

    def persist(self, context, item, output) -> list[str]:
        track_id, contours = output
        stored = self.batch.commit(
            lambda: store_track_contours(
                self.batch.connection, track_id, self.run_id, b"", usable=contours
            )
        )
        return ["melody_contours"] * stored


class FingerprintPhase(_SongPhase):
    """Chromaprints the selected songs and groups each with a recording it matches."""

    name = "fingerprint"
    summary_key = "fingerprint"
    counters = ("fingerprinted",)
    activity = "Fingerprinting"
    failure_message = "Could not fingerprint"

    def process(self, context, items):
        for item in items:
            context.check()
            try:
                audio, track_id = self.batch.audio(item.song)
                fingerprint, measured, version = compute_chromaprint(audio)
                del audio
            except Exception as error:  # noqa: BLE001 - one song's audio or fpcalc
                self.log_failure(item)
                yield ItemResult.failed(item, _failure(self.failure_message, error))
                continue
            duration = measured or item.song.duration
            output = (track_id, fingerprint, duration, version)
            yield ItemResult(item=item, outcome="fingerprinted", output=output)

    def persist(self, context, item, output) -> tuple[str, ...]:
        connection = self.batch.connection

        def store():
            with connection.transaction():
                return store_fingerprint(connection, *output)

        match = self.batch.commit(store)
        return ("recording_matches",) if match.get("matched") else ()


class RecordingIndexPhase(_SongPhase):
    """Indexes the selected songs for recording search with the configured encoder."""

    name = "recording_fingerprint"
    summary_key = "recording_fingerprint"
    counters = ("recording_fingerprinted",)
    activity = "Recording indexing finished for"
    failure_message = "Could not index recording"

    def plan(self, context):
        items = super().plan(context)
        if items:
            from worker.recordings.encoder import config_from_env

            self.config = config_from_env()
            if self.config is None:
                raise ValueError("Recording encoder configuration disappeared during analysis")
        return items

    def process(self, context, items):
        from worker.recordings.search import encode_fingerprints, needs_fingerprints

        for item in items:
            context.check()
            try:
                audio, track_id = self.batch.audio(item.song)
                if not needs_fingerprints(self.batch.connection, track_id, audio, self.config):
                    yield ItemResult(item=item, outcome=None)
                    continue
                fingerprints = encode_fingerprints(audio, self.config, context.check)
                del audio
            except Exception as error:  # noqa: BLE001 - one song's audio or encoding
                self.batch.connection.rollback()
                logger.error("Could not index recording for Navidrome song %s", item.key)
                yield ItemResult.failed(item, _failure(self.failure_message, error))
                continue
            yield ItemResult(item=item, outcome=None, output=(track_id, fingerprints))

    def persist(self, context, item, output) -> tuple[str, ...]:
        from worker.recordings.search import insert_fingerprints

        track_id, fingerprints = output
        stored = self.batch.commit(
            lambda: insert_fingerprints(self.batch.connection, track_id, fingerprints, self.config)
        )
        return ("recording_fingerprinted",) if stored else ()


class DescriptorPhase(_SongPhase):
    """Measures the selected songs' sound descriptors.

    A partial measurement is stored and counted as failed.
    """

    name = "audio_descriptors"
    summary_key = "audio_descriptors"
    counters = ("described_audio",)
    activity = "Measuring sound for"
    failure_message = "Could not analyze sound descriptors"

    def analyze(self, context, audio):
        return analyze_audio_descriptors(audio)

    def outcome(self, result):
        return "failed" if result.get("failed_components") else "described_audio"

    def persist(self, context, item, output):
        self.batch.commit(lambda: store_audio_descriptors(self.batch.connection, *output))


def ingest_navidrome(
    url: str,
    username: str,
    password: str,
    song_ids: list[str],
    progress: Callable[[dict[str, object]], None] | None = None,
    *,
    verify_audio_hashes: bool = True,
) -> IngestSummary:
    """Ingest a batch of Navidrome songs: identity, then each analysis the plan asks for.

    Phases, in order: identity; audio preparation; waveforms; visual features; MuQ-MuLan and
    MERT embeddings (each model loaded once); melody contours; chromaprints; recording index;
    sound descriptors. Each runs as a processor over the songs that need it.
    """
    report = progress or (lambda _: None)
    summary = IngestSummary(requested=len(song_ids))
    remote = current_remote()
    context = ProcessingContext(check_cancelled=get_check(), publish=report, remote=remote)

    def run(processor: Processor) -> None:
        counts = run_stage(processor, context).as_dict()
        for key, value in counts.items():
            setattr(summary, key, getattr(summary, key) + value)

    with (
        connect() as connection,
        NavidromeClient(url, username, password) as navidrome,
    ):
        configure_representations(connection)
        batch = IngestBatch(connection, navidrome, _library(connection, url), verify_audio_hashes)
        found = navidrome.tracks(song_ids)
        summary.discovered = len(found)
        # Selected sources require an identity check independently of artifact completeness.
        # Bind later consumers to these exact bytes and ids, even when another worker remaps
        # the source before this claim finishes.
        run(IdentityPhase(batch, found))
        songs = batch.songs
        plan = plan_audio(
            connection,
            batch.library_id,
            [song.id for song in songs],
            resolved_track_ids={key: value[0] for key, value in batch.bindings.items()},
        )
        report(
            {
                "phase": "planning",
                "message": "Processing plan ready",
                "completed": 0,
                "total": len(plan.download_external_ids),
                "unit": "tracks",
                "plan": {
                    "muq": len(plan.muq_external_ids),
                    "mert": len(plan.mert_external_ids),
                    "fingerprint": len(plan.fingerprint_external_ids),
                    "recording_fingerprint": len(plan.recording_fingerprint_external_ids),
                    "melody": len(plan.melody_external_ids),
                    "descriptors": len(plan.descriptor_external_ids),
                    "waveform": len(plan.waveform_external_ids),
                    "visual_features": len(plan.visual_feature_external_ids),
                },
            }
        )
        _prepare_audio(context, batch, plan)
        run(WaveformPhase(batch, plan.waveform_external_ids))
        run(VisualFeaturePhase(batch, plan.visual_feature_external_ids))
        models = int(plan.needs_muq) + int(plan.needs_mert)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = EmbeddingConfig.load()

        def muq(context):
            if context.remote is not None:
                return RemoteModel(MuQMuLanModel.name, config.muq_model_id, config.muq_revision)
            return MuQMuLanModel(config.muq_model_id, config.muq_revision, device)

        def mert(context):
            if context.remote is not None:
                return RemoteModel(MertModel.name, config.mert_model_id, config.mert_revision)
            return MertModel(config.mert_model_id, config.mert_revision, device)

        if plan.needs_muq:
            run(
                EmbeddingPhase(
                    batch,
                    plan.muq_external_ids,
                    "muq",
                    "Embedding semantics for",
                    muq,
                    ("Loading semantic model", 0, models),
                )
            )
        if plan.needs_mert:
            run(
                EmbeddingPhase(
                    batch,
                    plan.mert_external_ids,
                    "mert",
                    "Embedding acoustics for",
                    mert,
                    ("Loading acoustic model", int(plan.needs_muq), models),
                )
            )
        run(MelodyPhase(batch, plan.melody_external_ids))
        run(FingerprintPhase(batch, plan.fingerprint_external_ids))
        run(RecordingIndexPhase(batch, plan.recording_fingerprint_external_ids))
        run(DescriptorPhase(batch, plan.descriptor_external_ids))
        summary.already_linked = len(songs) - len(plan.download_external_ids)
        report(
            {
                "phase": "finalizing",
                "message": "Analysis phases complete",
                "completed": len(songs),
                "total": len(songs),
                "unit": "tracks",
            }
        )
    return summary
