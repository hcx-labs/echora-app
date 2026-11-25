"""Queries on the library's songs: visibility, sources and per-song analysis results.

Every function takes an open connection (``core.db.connection.connect``); the caller owns the
transaction. Rows come back as dicts typed below, in the stored shape.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, TypedDict, cast

from psycopg import Connection

from core.db.connection import fetch_all, fetch_one
from core.errors import NotFoundError


class AudioQuality(TypedDict):
    """The format of a song's source file."""

    codec: str | None
    content_type: str | None
    bit_rate_kbps: int | None
    bit_depth: int | None
    sample_rate_hz: int | None
    channels: int | None


class AudioDescriptors(TypedDict):
    """A song's tempo, key, loudness and other descriptors at one revision."""

    revision: str
    status: str
    descriptors: dict[str, Any] | None
    created_at: datetime


class VocalActivity(TypedDict):
    """Where a song has vocals, from a voice classification run."""

    activity: dict[str, Any]
    run_id: uuid.UUID


class Waveform(TypedDict):
    """A song's waveform peaks at one revision."""

    revision: str
    duration_seconds: float
    peaks: list[float]


class VisualFeatures(TypedDict):
    """A song's visualizer features at one revision."""

    revision: str
    status: str
    duration_seconds: float | None
    hop_seconds: float | None
    features: dict[str, Any] | None
    created_at: datetime


class Contour(TypedDict):
    """A song's melody contour (pitch and voicing per hop)."""

    source: str
    pitch: list[float]
    voiced: list[bool]
    hop_seconds: float


class RecordingGroup(TypedDict):
    """Songs that are the same recording."""

    id: uuid.UUID
    status: str
    canonical_track_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class RecordingGroupMember(TypedDict):
    """A song of a recording group."""

    id: uuid.UUID
    title: str | None
    artist: str | None
    album: str | None
    duration_seconds: float | None
    confidence: float | None
    membership_status: str
    source_ids: list[str]


class RecordingMatchEvidence(TypedDict):
    """One matcher decision between two songs."""

    id: uuid.UUID
    left_track_id: uuid.UUID
    right_track_id: uuid.UUID
    decision: str
    chromaprint_score: float | None
    duration_delta_seconds: float | None
    semantic_similarity: float | None
    acoustic_similarity: float | None
    matcher_revision: str
    created_at: datetime


def is_visible(db: Connection, user_id: uuid.UUID, track_id: uuid.UUID) -> bool:
    """Whether the song is in the user's library."""
    return (
        fetch_one(
            db,
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user_id, track_id),
        )
        is not None
    )


def require_visible(db: Connection, user_id: uuid.UUID, track_id: uuid.UUID, detail: str) -> None:
    """Raise ``NotFoundError(detail)`` unless the song is in the user's library."""
    if not is_visible(db, user_id, track_id):
        raise NotFoundError(detail)


def audio_quality(db: Connection, user_id: uuid.UUID, track_id: uuid.UUID) -> AudioQuality | None:
    """The format of the user's copy of the song, from its Navidrome source."""
    return cast(
        AudioQuality | None,
        fetch_one(
            db,
            """SELECT ts.source_data->>'suffix' AS codec,
                      ts.source_data->>'contentType' AS content_type,
                      (ts.source_data->>'bitRate')::integer AS bit_rate_kbps,
                      (ts.source_data->>'bitDepth')::integer AS bit_depth,
                      (ts.source_data->>'samplingRate')::integer AS sample_rate_hz,
                      (ts.source_data->>'channelCount')::integer AS channels
               FROM user_track_links link
               JOIN track_sources ts ON ts.track_id=link.track_id AND ts.library_id=link.library_id
                                    AND ts.external_id=link.external_id
               WHERE link.user_id=%s AND link.track_id=%s
               ORDER BY ts.id LIMIT 1""",
            (user_id, track_id),
        ),
    )


def audio_descriptors(
    db: Connection, track_id: uuid.UUID, revision: str
) -> AudioDescriptors | None:
    """The song's audio descriptors at ``revision``."""
    return cast(
        AudioDescriptors | None,
        fetch_one(
            db,
            """SELECT revision, status, descriptors, created_at FROM track_audio_descriptors
               WHERE track_id=%s AND revision=%s""",
            (track_id, revision),
        ),
    )


def vocal_activity(db: Connection, track_id: uuid.UUID) -> VocalActivity | None:
    """The song's latest vocal activity from a current voice classification run."""
    return cast(
        VocalActivity | None,
        fetch_one(
            db,
            """SELECT activity.activity, activity.run_id
               FROM track_vocal_activity activity
               JOIN current_analysis_runs ar ON ar.id=activity.run_id
               WHERE activity.track_id=%s AND ar.kind='voice_classification'
               ORDER BY activity.created_at DESC LIMIT 1""",
            (track_id,),
        ),
    )


def waveform(db: Connection, track_id: uuid.UUID, revision: str) -> Waveform | None:
    """The song's waveform peaks at ``revision``."""
    return cast(
        Waveform | None,
        fetch_one(
            db,
            "SELECT revision, duration_seconds, peaks FROM track_waveforms WHERE track_id=%s AND revision=%s",
            (track_id, revision),
        ),
    )


def visual_features(db: Connection, track_id: uuid.UUID, revision: str) -> VisualFeatures | None:
    """The song's cached visualizer features at ``revision``."""
    return cast(
        VisualFeatures | None,
        fetch_one(
            db,
            """SELECT revision, status, duration_seconds, hop_seconds, features, created_at
               FROM track_visual_features WHERE track_id=%s AND revision=%s""",
            (track_id, revision),
        ),
    )


def preview_contour(
    db: Connection, track_id: uuid.UUID, model_revision: str | None = None
) -> Contour | None:
    """The song's best melody contour: full mix before vocals, then the newest run.

    Runs that are still going count. ``model_revision`` limits it to runs of that revision.
    """
    revision = "AND ar.model_revision=%s" if model_revision is not None else ""
    params = (track_id, model_revision) if model_revision is not None else (track_id,)
    return cast(
        Contour | None,
        fetch_one(
            db,
            f"""SELECT mc.source, mc.pitch, mc.voiced, mc.hop_seconds
                FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
                WHERE mc.track_id=%s {revision} AND ar.status IN ('complete','running')
                ORDER BY CASE mc.source WHEN 'full-mix' THEN 0 WHEN 'vocals' THEN 1 ELSE 2 END,
                         ar.created_at DESC LIMIT 1""",
            params,
        ),
    )


def recording_group(db: Connection, track_id: uuid.UUID) -> RecordingGroup | None:
    """The recording group the song belongs to."""
    return cast(
        RecordingGroup | None,
        fetch_one(
            db,
            """SELECT rg.id, rg.status, rg.canonical_track_id, rg.created_at, rg.updated_at
               FROM recording_group_members member JOIN recording_groups rg ON rg.id=member.group_id
               WHERE member.track_id=%s""",
            (track_id,),
        ),
    )


def is_fingerprinted(db: Connection, track_id: uuid.UUID) -> bool:
    """Whether the song has a recording fingerprint."""
    return (
        fetch_one(db, "SELECT 1 FROM track_fingerprints WHERE track_id=%s", (track_id,)) is not None
    )


def recording_group_members(db: Connection, group_id: uuid.UUID) -> list[RecordingGroupMember]:
    """The group's songs by title, each with its source song ids."""
    return cast(
        list[RecordingGroupMember],
        fetch_all(
            db,
            """SELECT t.id, t.title, t.artist, t.album, t.duration_seconds, member.confidence,
                      member.membership_status, array_agg(DISTINCT ts.external_id) AS source_ids
               FROM recording_group_members member JOIN tracks t ON t.id=member.track_id
               LEFT JOIN track_sources ts ON ts.track_id=t.id WHERE member.group_id=%s
               GROUP BY t.id, member.confidence, member.membership_status ORDER BY t.title""",
            (group_id,),
        ),
    )


def recording_match_evidence(
    db: Connection, track_ids: list[uuid.UUID]
) -> list[RecordingMatchEvidence]:
    """Matcher decisions between any two of the songs, newest first."""
    return cast(
        list[RecordingMatchEvidence],
        fetch_all(
            db,
            """SELECT id, left_track_id, right_track_id, decision, chromaprint_score,
                      duration_delta_seconds, semantic_similarity, acoustic_similarity,
                      matcher_revision, created_at FROM recording_match_evidence
               WHERE left_track_id=ANY(%s) AND right_track_id=ANY(%s) ORDER BY created_at DESC""",
            (track_ids, track_ids),
        ),
    )


class AudioProfile(TypedDict, total=False):
    """A song's current audio profile and its profile and source runs.

    The ``current_audio_profiles`` columns plus ``profile_revision``, ``profile_config``,
    ``profile_created_at``, ``source_model_revision`` and ``source_config``.
    """

    id: uuid.UUID
    profile_run_id: uuid.UUID
    source_run_id: uuid.UUID
    model_name: str
    profile_revision: str | None
    profile_config: dict[str, Any] | None
    profile_created_at: datetime
    source_model_revision: str | None
    source_config: dict[str, Any] | None
    timestamps_exact: bool | None


def source_data(db: Connection, user_id: uuid.UUID, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The Navidrome song record of the user's copy of the song (``{"source_data": ...}``)."""
    return fetch_one(
        db,
        """SELECT ts.source_data FROM user_track_links link
           JOIN track_sources ts ON ts.track_id=link.track_id AND ts.library_id=link.library_id
                                AND ts.external_id=link.external_id
           WHERE link.user_id=%s AND link.track_id=%s ORDER BY ts.id LIMIT 1""",
        (user_id, track_id),
    )


def latest_descriptors(db: Connection, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The song's newest complete or partial descriptors, any revision (``{"descriptors": ...}``)."""
    return fetch_one(
        db,
        """SELECT descriptors FROM track_audio_descriptors
           WHERE track_id=%s AND status IN ('complete', 'partial')
           ORDER BY created_at DESC LIMIT 1""",
        (track_id,),
    )


def audio_profile(
    db: Connection, user_id: uuid.UUID, track_id: uuid.UUID, model: str
) -> AudioProfile | None:
    """The user's song's newest audio profile from ``model``'s embeddings."""
    return cast(
        AudioProfile | None,
        fetch_one(
            db,
            """SELECT tap.*, profile_run.model_revision AS profile_revision,
                      profile_run.config AS profile_config, profile_run.created_at AS profile_created_at,
                      source_run.model_revision AS source_model_revision,
                      source_run.config AS source_config
               FROM current_audio_profiles tap
               JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
               JOIN analysis_runs source_run ON source_run.id=tap.source_run_id
               WHERE tap.track_id=%s AND tap.model_name=%s
                 AND EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.user_id=%s AND utl.track_id=tap.track_id)
               ORDER BY profile_run.created_at DESC, tap.created_at DESC LIMIT 1""",
            (track_id, model, user_id),
        ),
    )


def profile_segments(db: Connection, profile_id: uuid.UUID) -> list[dict[str, Any]]:
    """The profile's temporal segments in order."""
    return fetch_all(
        db,
        """SELECT segment_index, start_seconds, end_seconds, cohesion,
                  representative_window_index
           FROM audio_temporal_segments
           WHERE profile_id=%s ORDER BY segment_index""",
        (profile_id,),
    )


def profile_modes(db: Connection, profile_id: uuid.UUID) -> list[dict[str, Any]]:
    """The profile's modes in order, each with its ``intervals``."""
    modes = fetch_all(
        db,
        """SELECT am.id, am.mode_index, am.duration_weight, am.cohesion,
                  am.representative_window_index
           FROM audio_modes am
           WHERE am.profile_id=%s ORDER BY am.mode_index""",
        (profile_id,),
    )
    for mode in modes:
        mode["intervals"] = fetch_all(
            db,
            """SELECT start_seconds, end_seconds FROM audio_mode_intervals
               WHERE mode_id=%s ORDER BY interval_index""",
            (mode["id"],),
        )
        mode.pop("id")
    return modes
