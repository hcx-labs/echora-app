"""The stages of an analysis batch: each runs one kind of processing over the batch's songs.

A stage returns the part of the batch summary it owns (``{"lyrics": {...}}``, or top-level keys
for the library stage). Stage modules are imported when the stage runs, so a worker loads only the
models its operations use.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.compute import location
from core.db.connection import connect
from core.library.membership import attach_user_library, user_audio_track_ids
from worker.library.sources import source_rows
from worker.processing import ProcessingContext, Processor, run_stage
from worker.remote.session import current as remote_session

logger = logging.getLogger(__name__)

Credentials = tuple[str, str, str]


@dataclass
class Batch:
    """One claimed batch: whose songs, from which library, and the job's progress and checks."""

    user_id: str
    track_ids: list[str]
    payload: dict[str, Any]
    report: Callable[[dict[str, Any]], None]
    check: Callable[[], None]
    credentials: Credentials | None = None
    prepared: dict[str, Any] = field(default_factory=dict)
    """What prepare hooks produced for later stages, by stage key."""

    def context(self) -> ProcessingContext:
        """The processing context of this batch, for processors."""
        return ProcessingContext(
            check_cancelled=self.check,
            publish=self.report,
            location=location(),  # type: ignore[arg-type]
            remote=remote_session(),
            user_id=self.user_id,
            library=self.credentials,
            track_ids=tuple(self.track_ids),
            options=self.payload,
            resources=self.prepared,
        )

    @property
    def library(self) -> Credentials:
        """The Navidrome connection; operations without one never run library stages."""
        if self.credentials is None:
            raise ValueError("This batch has no library connection")
        return self.credentials

    @property
    def url(self) -> str:
        """The library's Navidrome URL."""
        return self.library[0]


@dataclass(frozen=True)
class Stage:
    """One stage: its summary key, its work, and optional work before any model loads."""

    key: str
    run: Callable[[Batch], dict[str, Any]]
    prepare: Callable[[Batch], Any] | None = None
    """Runs before the batch opens remote computation (no GPU waits for it); best effort."""


def processor_stage(key: str, build: Callable[[], Processor]) -> Stage:
    """A stage that runs one processor; its summary is stored under ``key``."""

    def run(batch: Batch) -> dict[str, Any]:
        return {key: run_stage(build(), batch.context()).as_dict()}

    return Stage(key, run)


def _add_links(user_id, url: str, ids: list[str]) -> int:
    from core.library.visibility import update_memberships

    # Never reconcile a batch: removing absent IDs would delete sibling batches.
    from core.planning import store

    with connect() as connection:
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/"))
        library_id = store.library_with_namespace(connection, namespace)
        if library_id is None:
            return 0
        return update_memberships(connection, library_id, user_id, ids)["linked"]


def profiles(batch: Batch, ids) -> dict[str, Any]:
    """Audio profiles of canonical tracks for every supported embedding model."""
    from worker.profiles.pipeline import SUPPORTED_PROFILE_MODELS, AudioProfileProcessor

    track_ids = [uuid.UUID(str(v)) for v in ids]
    models = {
        name: run_stage(AudioProfileProcessor(track_ids, name), batch.context()).as_dict()
        for name in SUPPORTED_PROFILE_MODELS
    }
    return {
        "models": models,
        "profiled": sum(v["profiled"] for v in models.values()),
        "failed": sum(v["failed"] for v in models.values()),
    }


# Library: ingest, membership and profiles.


def _library(batch: Batch) -> dict[str, Any]:
    from worker.library.ingest import ingest_navidrome

    summary: dict[str, Any] = asdict(
        ingest_navidrome(
            *batch.library,
            batch.track_ids,
            batch.report,
            verify_audio_hashes=batch.payload.get("verify_audio_hashes", True),
        )
    )
    batch.check()
    attach_user_library(batch.user_id, batch.url)
    summary["linked"] = _add_links(batch.user_id, batch.url, batch.track_ids)
    canonical = list({row[1] for row in source_rows(batch.user_id, batch.url, batch.track_ids)})
    summary["audio_profiles"] = profiles(batch, canonical)
    return summary


def _audio_profiles(batch: Batch) -> dict[str, Any]:
    allowed = {str(v) for v in user_audio_track_ids(batch.user_id)}
    return profiles(batch, [v for v in batch.track_ids if str(v) in allowed])


# Lyrics and what is derived from them.


def _lyrics(batch: Batch) -> dict[str, Any]:
    from worker.lyrics.pipeline import lyrics_stage

    return {"lyrics": lyrics_stage(batch.context())}


def _theme_processor() -> Processor:
    from worker.themes.pipeline import ThemeProcessor

    return ThemeProcessor()


def _karaoke_processor() -> Processor:
    from worker.kara.pipeline import KaraokeProcessor

    return KaraokeProcessor()


def _translation_processor() -> Processor:
    from worker.translation.pipeline import TranslationProcessor

    return TranslationProcessor()


# Audio analysis.


def _voice_processor() -> Processor:
    from worker.voice.pipeline import VoiceProcessor

    return VoiceProcessor()


def _prepare_artwork_prompts(batch: Batch) -> Any:
    from worker.artwork.pipeline import prepare_prompts

    return prepare_prompts(batch.library, batch.track_ids, progress=batch.report, check=batch.check)


def _motion_artwork(batch: Batch) -> dict[str, Any]:
    from worker.artwork.pipeline import MotionArtworkProcessor

    processor = MotionArtworkProcessor(prompts=batch.prepared.get("motion_artwork"))
    summary = run_stage(processor, batch.context()).as_dict()
    # Off at this location: the batch summary has no motion artwork key.
    return {"motion_artwork": summary} if processor.enabled else {}


def _fingerprint_processor() -> Processor:
    from worker.library.recordings import FingerprintProcessor

    return FingerprintProcessor()


def _melody_processor() -> Processor:
    from worker.melody.corpus import MelodyCorpusProcessor

    return MelodyCorpusProcessor()


def _semantic_fusion(batch: Batch) -> dict[str, Any]:
    from worker.fusion.pipeline import build_semantic_fusion

    batch.report(
        {
            "phase": "building",
            "message": "Building semantic fusion vectors",
            "completed": 0,
            "total": 1,
            "unit": "builds",
        }
    )
    return build_semantic_fusion(progress=batch.report)


LIBRARY = Stage("library", _library)
AUDIO_PROFILES = Stage("audio_profiles", _audio_profiles)
LYRICS = Stage("lyrics", _lyrics)
TRANSLATIONS = processor_stage("translations", _translation_processor)
THEMES = processor_stage("themes", _theme_processor)
KARAOKE = processor_stage("karaoke", _karaoke_processor)
VOICE = processor_stage("voice", _voice_processor)
# Last in a sync, so earlier stages have released their models before ComfyUI needs the GPU.
# Prompts come first, so no GPU (here or on Modal) idles while External AI writes them.
MOTION_ARTWORK = Stage("motion_artwork", _motion_artwork, prepare=_prepare_artwork_prompts)
# The summary's counters sit at the top level of a recordings batch, as they always have.
RECORDINGS = Stage(
    "recordings", lambda batch: run_stage(_fingerprint_processor(), batch.context()).as_dict()
)
# Like recordings, a hum corpus batch's counters sit at the top level.
MELODY = Stage("melody", lambda batch: run_stage(_melody_processor(), batch.context()).as_dict())
SEMANTIC_FUSION = Stage("semantic_fusion", _semantic_fusion)
