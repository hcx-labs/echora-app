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

import psycopg

from core.db.config import require_database_url
from core.integrations.navidrome import NavidromeClient
from core.library.membership import attach_user_library, user_audio_track_ids

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


def connect() -> psycopg.Connection:
    """A connection to the Echora database."""
    return psycopg.connect(require_database_url())


def source_rows(user_id, url: str, ids: list[str] | None = None) -> list[tuple[str, uuid.UUID]]:
    """``(external_id, track_id)`` of the user's songs in this library, optionally only ``ids``."""
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT DISTINCT ts.external_id, ts.track_id FROM track_sources ts
               JOIN libraries l ON l.id=ts.library_id
               JOIN user_track_links u ON u.library_id=ts.library_id AND u.track_id=ts.track_id
                 AND u.external_id=ts.external_id
               WHERE u.user_id=%s AND l.root_path=%s AND ts.source_type='subsonic'
                 AND (%s::text[] IS NULL OR ts.external_id=ANY(%s)) ORDER BY ts.external_id""",
            (user_id, url.rstrip("/"), ids, ids),
        )
        return cursor.fetchall()


def _add_links(user_id, url: str, ids: list[str]) -> int:
    from core.library.visibility import update_memberships

    # Never reconcile a batch: removing absent IDs would delete sibling batches.
    with connect() as connection, connection.cursor() as cursor:
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/"))
        cursor.execute("SELECT id FROM libraries WHERE namespace=%s", (namespace,))
        library = cursor.fetchone()
        if library is None:
            return 0
        return update_memberships(connection, library[0], user_id, ids)["linked"]


def profiles(ids, report) -> dict[str, Any]:
    """Audio profiles of canonical tracks for every supported embedding model."""
    from worker.profiles.pipeline import SUPPORTED_PROFILE_MODELS, build_audio_profiles

    models = {
        name: build_audio_profiles([uuid.UUID(str(v)) for v in ids], report, model_name=name)
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
    summary["audio_profiles"] = profiles(canonical, batch.report)
    return summary


def _audio_profiles(batch: Batch) -> dict[str, Any]:
    allowed = {str(v) for v in user_audio_track_ids(batch.user_id)}
    return profiles([v for v in batch.track_ids if str(v) in allowed], batch.report)


# Lyrics and what is derived from them.


def _lyrics(batch: Batch) -> dict[str, Any]:
    from worker.lyrics.pipeline import backfill_lyrics

    return {
        "lyrics": backfill_lyrics(
            *batch.library,
            progress=batch.report,
            external_ids=batch.track_ids,
            only_missing=True,
            refresh_existing=batch.payload.get("refresh_lyrics", False),
        )
    }


def _translations(batch: Batch) -> dict[str, Any]:
    from worker.translation.pipeline import backfill_translations

    return {
        "translations": backfill_translations(
            batch.user_id, batch.url, batch.track_ids, check=batch.check, progress=batch.report
        )
    }


def _themes(batch: Batch) -> dict[str, Any]:
    from worker.themes.pipeline import backfill_themes

    return {
        "themes": backfill_themes(
            batch.user_id, batch.url, batch.track_ids, check=batch.check, progress=batch.report
        )
    }


def _karaoke(batch: Batch) -> dict[str, Any]:
    from worker.kara.pipeline import backfill_karaoke

    return {
        "karaoke": backfill_karaoke(
            *batch.library, progress=batch.report, external_ids=batch.track_ids
        )
    }


# Audio analysis.


def _voice(batch: Batch) -> dict[str, Any]:
    from worker.voice.pipeline import backfill_voice

    return {
        "voice": backfill_voice(*batch.library, progress=batch.report, external_ids=batch.track_ids)
    }


def _prepare_artwork_prompts(batch: Batch) -> Any:
    from worker.artwork.pipeline import prepare_prompts

    return prepare_prompts(batch.library, batch.track_ids, progress=batch.report, check=batch.check)


def _motion_artwork(batch: Batch) -> dict[str, Any]:
    from worker.artwork.pipeline import render_batch

    artwork = render_batch(
        batch.library,
        batch.user_id,
        batch.track_ids,
        progress=batch.report,
        check=batch.check,
        prompts=batch.prepared.get("motion_artwork"),
    )
    return {} if artwork is None else {"motion_artwork": artwork}


def _recordings(batch: Batch) -> dict[str, Any]:
    from worker.library.fingerprints import store_and_match_fingerprint

    summary = {"fingerprinted": 0, "matched": 0, "failed": 0}
    with (
        connect() as connection,
        connection.cursor() as cursor,
        NavidromeClient(*batch.library) as client,
    ):
        for external_id, track_id in source_rows(batch.user_id, batch.url, batch.track_ids):
            batch.check()
            cursor.execute(
                """SELECT duration_seconds FROM tracks WHERE id=%s
                   AND NOT EXISTS (SELECT 1 FROM track_fingerprints WHERE track_id=%s)""",
                (track_id, track_id),
            )
            row = cursor.fetchone()
            if not row:
                continue
            try:
                result = store_and_match_fingerprint(
                    connection, track_id, client.audio_bytes(external_id), float(row[0] or 0)
                )
                connection.commit()
                summary["fingerprinted"] += 1
                summary["matched"] += int(bool(result.get("matched")))
            except Exception:
                connection.rollback()
                summary["failed"] += 1
            batch.report({"phase": "fingerprint", "summary": dict(summary)})
    return summary


def _melody(batch: Batch) -> dict[str, Any]:
    from worker.media.preprocessing import prepare_audio
    from worker.melody.index import create_sync_run, store_track_contours

    corpus_id = batch.payload["corpus_id"]
    summary = {"tracks": 0, "contours": 0, "failed": 0}
    with (
        connect() as connection,
        connection.cursor() as cursor,
        NavidromeClient(*batch.library) as client,
    ):
        cursor.execute(
            "SELECT id FROM hum_corpora WHERE id=%s AND user_id=%s", (corpus_id, batch.user_id)
        )
        if not cursor.fetchone():
            raise ValueError("Unknown hum corpus")
        run_id = create_sync_run(connection)
        cursor.execute(
            "UPDATE hum_corpora SET run_id=%s WHERE id=%s AND user_id=%s",
            (run_id, corpus_id, batch.user_id),
        )
        connection.commit()
        for external_id, track_id in source_rows(batch.user_id, batch.url, batch.track_ids):
            batch.check()
            try:
                cursor.execute(
                    "SELECT count(*) FROM melody_contours WHERE track_id=%s AND run_id=%s",
                    (track_id, run_id),
                )
                if not cursor.fetchone()[0]:
                    batch.report(
                        {"phase": "preprocess", "message": "Preparing shared melody audio"}
                    )
                    audio = client.audio_bytes(external_id)
                    prepare_audio(
                        audio, mono_rates=(44100,), stereo=True, melody=True, check=batch.check
                    )
                    batch.report({"phase": "melody-index", "message": "Extracting melody contours"})
                    summary["contours"] += store_track_contours(connection, track_id, run_id, audio)
                    del audio
                cursor.execute(
                    """INSERT INTO hum_corpus_tracks(corpus_id,track_id) VALUES (%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (corpus_id, track_id),
                )
                connection.commit()
                summary["tracks"] += 1
            except Exception:
                connection.rollback()
                summary["failed"] += 1
            batch.report({"phase": "melody-index", "summary": dict(summary)})
    return summary


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
TRANSLATIONS = Stage("translations", _translations)
THEMES = Stage("themes", _themes)
KARAOKE = Stage("karaoke", _karaoke)
VOICE = Stage("voice", _voice)
# Last in a sync, so earlier stages have released their models before ComfyUI needs the GPU.
# Prompts come first, so no GPU (here or on Modal) idles while External AI writes them.
MOTION_ARTWORK = Stage("motion_artwork", _motion_artwork, prepare=_prepare_artwork_prompts)
RECORDINGS = Stage("recordings", _recordings)
MELODY = Stage("melody", _melody)
SEMANTIC_FUSION = Stage("semantic_fusion", _semantic_fusion)
