"""Plans which songs of a batch need each analysis, from what is already stored."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

import psycopg

from core.embeddings.config import EmbeddingConfig
from core.features import feature_enabled
from core.planning import store
from core.revisions import MELODY_CONTOUR_REVISION
from worker.library.descriptors import DESCRIPTOR_REVISION
from worker.library.visual_features import VISUAL_FEATURE_REVISION
from worker.library.waveforms import WAVEFORM_REVISION


@dataclass(frozen=True)
class ProcessingPlan:
    """The songs that need lyrics and karaoke work."""

    lyrics_external_ids: tuple[str, ...] = ()
    karaoke_external_ids: tuple[str, ...] = ()

    @property
    def needs_bge(self) -> bool:
        return bool(self.lyrics_external_ids)

    @property
    def needs_fa_kara(self) -> bool:
        return bool(self.karaoke_external_ids)

    @property
    def empty(self) -> bool:
        return not self.needs_bge and not self.needs_fa_kara


@dataclass(frozen=True)
class AudioProcessingPlan:
    """The songs (external ids) that need each audio analysis."""

    muq_external_ids: frozenset[str]
    mert_external_ids: frozenset[str]
    fingerprint_external_ids: frozenset[str]
    melody_external_ids: frozenset[str]
    descriptor_external_ids: frozenset[str] = frozenset()

    waveform_external_ids: frozenset[str] = frozenset()
    visual_feature_external_ids: frozenset[str] = frozenset()
    recording_fingerprint_external_ids: frozenset[str] = frozenset()

    @property
    def needs_muq(self) -> bool:
        return bool(self.muq_external_ids)

    @property
    def needs_mert(self) -> bool:
        return bool(self.mert_external_ids)

    @property
    def needs_melody(self) -> bool:
        return bool(self.melody_external_ids)

    @property
    def download_external_ids(self) -> frozenset[str]:
        return (
            self.muq_external_ids
            | self.mert_external_ids
            | self.fingerprint_external_ids
            | self.melody_external_ids
            | self.descriptor_external_ids
            | self.waveform_external_ids
            | self.visual_feature_external_ids
            | self.recording_fingerprint_external_ids
        )


@dataclass(frozen=True)
class AudioPrerequisites:
    """Exact decoded formats shared by the pending mix-analysis tasks."""

    mono_rates: tuple[int, ...] = ()
    stereo_rates: tuple[int, ...] = ()
    melody: bool = False


def audio_prerequisites(plan: AudioProcessingPlan, external_id: str) -> AudioPrerequisites:
    mono, stereo = set(), set()
    if external_id in plan.recording_fingerprint_external_ids:
        mono.add(8_000)
    if external_id in plan.muq_external_ids | plan.mert_external_ids:
        mono.add(24_000)
    if external_id in plan.melody_external_ids:
        mono.add(44_100)
        stereo.add(44_100)
    if external_id in plan.descriptor_external_ids:
        stereo.add(44_100)
    if external_id in plan.waveform_external_ids:
        stereo.add(24_000)
    if external_id in plan.visual_feature_external_ids:
        mono.add(22_050)
    # Chromaprint decodes independently; changing its input may change fingerprints.
    return AudioPrerequisites(
        tuple(sorted(mono)), tuple(sorted(stereo)), external_id in plan.melody_external_ids
    )


def resolve_library_id(connection: psycopg.Connection, url: str) -> uuid.UUID:
    """Resolve an existing library, never falling back to unscoped processing."""
    rows = store.libraries_at(connection, url)
    if len(rows) != 1:
        raise ValueError("Expected exactly one library for the source URL")
    return rows[0]


def plan_lyrics(
    connection: psycopg.Connection,
    external_ids: Iterable[str] | None = None,
    library_id: uuid.UUID | None = None,
) -> ProcessingPlan:
    revision = EmbeddingConfig.load().lyrics_revision
    ids = tuple(store.songs_needing_lyrics(connection, external_ids, library_id, revision))
    return ProcessingPlan(lyrics_external_ids=ids)


def plan_karaoke(
    connection: psycopg.Connection,
    pipeline_revision: str,
    external_ids: Iterable[str] | None = None,
    model_revision: str | None = None,
    library_id: uuid.UUID | None = None,
    realign_outdated: bool | None = None,
) -> ProcessingPlan:
    """Songs to align.

    ``realign_outdated`` (default: the administrator's setting) also plans alignments of another
    pipeline or model revision; songs with a karaoke resync request are always planned.
    """
    if not feature_enabled(connection, "karaoke"):
        return ProcessingPlan()
    if realign_outdated is None:
        from core.settings.store import karaoke_realign_outdated

        realign_outdated = karaoke_realign_outdated(connection)
    ids = tuple(
        store.songs_needing_karaoke(
            connection,
            external_ids,
            library_id,
            pipeline_revision,
            model_revision,
            realign_outdated,
        )
    )
    return ProcessingPlan(karaoke_external_ids=ids)


def plan_audio(
    connection: psycopg.Connection,
    library_id,
    external_ids: Iterable[str],
    *,
    resolved_track_ids: dict | None = None,
) -> AudioProcessingPlan:
    ids = list(external_ids)
    revisions = {
        "muq": EmbeddingConfig.load().muq_revision,
        "mert": EmbeddingConfig.load().mert_revision,
        "melody": MELODY_CONTOUR_REVISION,
        "descriptors": DESCRIPTOR_REVISION,
        "waveform": WAVEFORM_REVISION,
        "visual_features": VISUAL_FEATURE_REVISION,
    }
    hum_enabled = feature_enabled(connection, "hum")
    rows = store.audio_artifacts(connection, library_id, ids, resolved_track_ids, revisions)
    from worker.recordings.encoder import config_from_env

    recording_config = config_from_env()
    recording_ids = frozenset()
    if recording_config is not None:
        recording_ids = frozenset(
            store.songs_needing_recording_fingerprints(
                connection, library_id, ids, resolved_track_ids, recording_config.representation_id
            )
        )

    def missing(artifact: str) -> frozenset[str]:
        return frozenset(str(row["external_id"]) for row in rows if not row[artifact])

    return AudioProcessingPlan(
        missing("has_muq"),
        missing("has_mert"),
        missing("has_fingerprint"),
        missing("has_melody") if hum_enabled else frozenset(),
        missing("has_descriptors"),
        missing("has_waveform"),
        missing("has_visual_features"),
        recording_ids,
    )
