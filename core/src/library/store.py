"""Queries on the library's songs: visibility, sources and per-song analysis results.

Every function takes an open connection (``core.db.connection.connect``); the caller owns the
transaction. Rows come back as dicts typed below, in the stored shape.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, TypedDict, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

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


def library_songs(db: Connection, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """Every song of the user's library with its Navidrome source id and cover."""
    return fetch_all(
        db,
        """SELECT t.id::text AS id, t.title, t.artist, t.album, t.duration_seconds,
                  source.external_id AS source_id, source.cover_art
           FROM tracks t
           LEFT JOIN LATERAL (
             SELECT ts.external_id, ts.source_data->>'coverArt' AS cover_art
             FROM track_sources ts
             JOIN user_track_links links
               ON links.user_id=%s AND links.library_id=ts.library_id
              AND links.track_id=ts.track_id AND links.external_id=ts.external_id
             WHERE ts.track_id=t.id AND ts.source_type='subsonic'
             ORDER BY ts.id LIMIT 1
           ) source ON true
           WHERE EXISTS (SELECT 1 FROM user_track_links utl WHERE utl.track_id=t.id AND utl.user_id=%s)""",
        (user_id, user_id),
    )


def save_artist_profile(
    db: Connection,
    name: str,
    model: str,
    corpus_hash: str,
    track_count: int,
    component_count: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Store an artist's profile for its corpus (replacing the payload); ``id``, ``created_at``."""
    return cast(
        dict[str, Any],
        fetch_one(
            db,
            """INSERT INTO artist_profiles
                 (artist_key, artist_name, model_name, corpus_hash, track_count, component_count, payload)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (artist_key, model_name, corpus_hash) DO UPDATE SET payload=EXCLUDED.payload
               RETURNING id, created_at""",
            (
                name.casefold(),
                name,
                model,
                corpus_hash,
                track_count,
                component_count,
                Jsonb(payload),
            ),
        ),
    )


def community_snapshot(db: Connection, snapshot_id: uuid.UUID) -> dict[str, Any] | None:
    """A stored community map (``payload``, ``created_at``)."""
    return fetch_one(
        db, "SELECT payload, created_at FROM community_snapshots WHERE id=%s", (snapshot_id,)
    )


def cached_community_snapshot(
    db: Connection, model: str, semantic_weight: float, corpus_hash: str, revision: int
) -> dict[str, Any] | None:
    """The community map stored for this corpus and algorithm (``id``, ``payload``, ``created_at``)."""
    return fetch_one(
        db,
        """SELECT id, payload, created_at FROM community_snapshots
           WHERE model_name=%s AND semantic_weight=%s AND corpus_hash=%s AND algorithm_revision=%s""",
        (model, semantic_weight, corpus_hash, revision),
    )


def community_snapshots(db: Connection, limit: int) -> list[dict[str, Any]]:
    """The newest community maps, without their payloads."""
    return fetch_all(
        db,
        """SELECT id, model_name, semantic_weight, corpus_hash, algorithm_revision,
                  track_count, parameters, metrics, created_at
           FROM community_snapshots ORDER BY created_at DESC LIMIT %s""",
        (limit,),
    )


def save_community_snapshot(
    db: Connection,
    model: str,
    semantic_weight: float,
    corpus_hash: str,
    revision: int,
    track_count: int,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Store a community map for its corpus and algorithm; ``id``, ``created_at``."""
    return cast(
        dict[str, Any],
        fetch_one(
            db,
            """INSERT INTO community_snapshots
                 (model_name, semantic_weight, corpus_hash, algorithm_revision, track_count, parameters, metrics, payload)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (model_name, semantic_weight, corpus_hash, algorithm_revision)
               DO UPDATE SET payload=EXCLUDED.payload, parameters=EXCLUDED.parameters, metrics=EXCLUDED.metrics
               RETURNING id, created_at""",
            (
                model,
                semantic_weight,
                corpus_hash,
                revision,
                track_count,
                Jsonb(parameters),
                Jsonb(metrics),
                Jsonb(payload),
            ),
        ),
    )


def songs_without_contour(db: Connection, revision: str) -> int:
    """Navidrome songs that have no melody contour at ``revision`` yet."""
    row = fetch_one(
        db,
        """SELECT count(DISTINCT ts.track_id) AS pending
           FROM track_sources ts
           WHERE ts.source_type='subsonic'
             AND NOT EXISTS (
               SELECT 1 FROM melody_contours mc
               JOIN analysis_runs ar ON ar.id=mc.run_id
               WHERE mc.track_id=ts.track_id
                 AND ar.model_name='melody_contour' AND ar.model_revision=%s
             )""",
        (revision,),
    )
    return int(row["pending"]) if row else 0


def processed_source_ids(db: Connection, namespace: uuid.UUID) -> set[str]:
    """Navidrome song ids of the library namespace that Echora has ingested."""
    rows = fetch_all(
        db,
        """SELECT ts.external_id FROM track_sources ts
           JOIN libraries l ON l.id=ts.library_id
           WHERE l.namespace=%s AND ts.source_type='subsonic'""",
        (namespace,),
    )
    return {str(row["external_id"]) for row in rows}


def songs_with_contours(db: Connection, user_id: uuid.UUID) -> int:
    """How many of the user's songs have a melody contour (hum search covers them)."""
    row = fetch_one(
        db,
        """SELECT count(DISTINCT mc.track_id) AS indexed_tracks
           FROM melody_contours mc
           WHERE EXISTS (SELECT 1 FROM user_track_links utl
                         WHERE utl.user_id=%s AND utl.track_id=mc.track_id)""",
        (user_id,),
    )
    return int(row["indexed_tracks"]) if row else 0


# Ingest: songs, their sources and libraries, and the per-song analysis results it stores.


def song_sources(
    db: Connection, user_id: uuid.UUID, library_root: str, external_ids: list[str] | None
) -> list[dict[str, Any]]:
    """``external_id`` and ``track_id`` of the user's songs in the library, optionally only some."""
    return fetch_all(
        db,
        """SELECT DISTINCT ts.external_id, ts.track_id FROM track_sources ts
           JOIN libraries l ON l.id=ts.library_id
           JOIN user_track_links u ON u.library_id=ts.library_id AND u.track_id=ts.track_id
             AND u.external_id=ts.external_id
           WHERE u.user_id=%s AND l.root_path=%s AND ts.source_type='subsonic'
             AND (%s::text[] IS NULL OR ts.external_id=ANY(%s)) ORDER BY ts.external_id""",
        (user_id, library_root, external_ids, external_ids),
    )


def library_id(db: Connection, root: str, namespace: uuid.UUID) -> uuid.UUID:
    """The id of the Navidrome library at ``root``, stored first when new."""
    row = fetch_one(
        db,
        """
        INSERT INTO libraries (name, root_path, namespace)
        VALUES ('Navidrome', %s, %s)
        ON CONFLICT (namespace) DO UPDATE SET root_path=EXCLUDED.root_path
        RETURNING id
        """,
        (root, namespace),
    )
    return cast(dict, row)["id"]


def source_track_id(db: Connection, library: uuid.UUID, external_id: str) -> uuid.UUID | None:
    """The song a Navidrome source is bound to."""
    row = fetch_one(
        db,
        "SELECT track_id FROM track_sources WHERE library_id=%s AND source_type='subsonic' AND external_id=%s",
        (library, external_id),
    )
    return row["track_id"] if row else None


def known_source(db: Connection, library: uuid.UUID, external_id: str) -> dict[str, Any] | None:
    """The bound song of a source: ``track_id`` and its ``audio_hash``."""
    return fetch_one(
        db,
        """SELECT ts.track_id, t.audio_hash FROM track_sources ts
        JOIN tracks t ON t.id=ts.track_id
        WHERE ts.library_id=%s AND ts.source_type='subsonic' AND ts.external_id=%s""",
        (library, external_id),
    )


def add_track(db: Connection, track_id: uuid.UUID, audio_hash: str, **fields: Any) -> bool:
    """Store a song by its audio hash unless one exists; whether it was new.

    ``fields`` are ``title``, ``artist``, ``album``, ``year``, ``duration_seconds``, ``genres``
    and ``metadata``.
    """
    return (
        fetch_one(
            db,
            """
            INSERT INTO tracks (id, audio_hash, title, artist, album, year, duration_seconds, genres, metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (audio_hash) DO NOTHING
            RETURNING id
            """,
            (
                track_id,
                audio_hash,
                fields["title"],
                fields["artist"],
                fields["album"],
                fields["year"],
                fields["duration_seconds"],
                fields["genres"],
                Jsonb(fields["metadata"]),
            ),
        )
        is not None
    )


def track_with_hash(db: Connection, audio_hash: str) -> uuid.UUID:
    """The song stored for an audio hash."""
    row = fetch_one(db, "SELECT id FROM tracks WHERE audio_hash=%s", (audio_hash,))
    return cast(dict, row)["id"]


def bind_source(
    db: Connection,
    library: uuid.UUID,
    track_id: uuid.UUID,
    relative_path: str | None,
    external_id: str,
    source_data: dict[str, Any],
) -> None:
    """Put the song in the library and bind the Navidrome source to it."""
    db.execute(
        "INSERT INTO library_tracks (library_id, track_id, relative_path) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
        (library, track_id, relative_path),
    )
    db.execute(
        """
        INSERT INTO track_sources (library_id, track_id, source_type, external_id, source_data)
        VALUES (%s,%s,'subsonic',%s,%s)
        ON CONFLICT (library_id, source_type, external_id)
        DO UPDATE SET track_id=EXCLUDED.track_id, source_data=EXCLUDED.source_data
        """,
        (library, track_id, external_id, Jsonb(source_data)),
    )


def refresh_source(
    db: Connection,
    library: uuid.UUID,
    external_id: str,
    source_data: dict[str, Any],
    track_id: uuid.UUID,
    **fields: Any,
) -> None:
    """Store a source's current Navidrome data and its song's tags.

    ``fields`` are ``title``, ``artist``, ``album``, ``year``, ``duration_seconds`` and ``genres``.
    """
    db.execute(
        """UPDATE track_sources SET source_data=%s
        WHERE library_id=%s AND source_type='subsonic' AND external_id=%s""",
        (Jsonb(source_data), library, external_id),
    )
    db.execute(
        """UPDATE tracks SET title=%s, artist=%s, album=%s, year=%s,
        duration_seconds=%s, genres=%s WHERE id=%s""",
        (
            fields["title"],
            fields["artist"],
            fields["album"],
            fields["year"],
            fields["duration_seconds"],
            fields["genres"],
            track_id,
        ),
    )


def mark_source_verified(
    db: Connection, library: uuid.UUID, external_id: str, track_id: uuid.UUID
) -> None:
    """The source's audio was hashed now and is the song's."""
    db.execute(
        """UPDATE track_sources SET audio_verified_at=now()
        WHERE library_id=%s AND source_type='subsonic' AND external_id=%s
            AND track_id=%s""",
        (library, external_id, track_id),
    )


def save_waveform(
    db: Connection, track_id: uuid.UUID, revision: str, duration: float, peaks: Any
) -> None:
    """Store the song's waveform peaks."""
    db.execute(
        """INSERT INTO track_waveforms (track_id, revision, duration_seconds, peaks)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (track_id) DO UPDATE SET revision=EXCLUDED.revision,
             duration_seconds=EXCLUDED.duration_seconds, peaks=EXCLUDED.peaks, created_at=now()""",
        (track_id, revision, duration, Jsonb(peaks)),
    )


def save_audio_descriptors(
    db: Connection, track_id: uuid.UUID, revision: str, status: str, descriptors: dict[str, Any]
) -> None:
    """Store the song's audio descriptors of a revision."""
    db.execute(
        """INSERT INTO track_audio_descriptors (track_id, revision, status, descriptors)
           VALUES (%s,%s,%s,%s)
           ON CONFLICT (track_id, revision) DO UPDATE SET
             status=EXCLUDED.status, descriptors=EXCLUDED.descriptors, created_at=now()""",
        (track_id, revision, status, Jsonb(descriptors)),
    )


def save_unsupported_visual_features(
    db: Connection, track_id: uuid.UUID, revision: str, duration: float, hop_seconds: float
) -> None:
    """Record that the song is too long for visual features."""
    db.execute(
        """INSERT INTO track_visual_features
             (track_id, revision, status, duration_seconds, hop_seconds, features)
           VALUES (%s, %s, 'unsupported', %s, %s, '{}'::jsonb)
           ON CONFLICT (track_id) DO UPDATE SET revision=EXCLUDED.revision,
             status=EXCLUDED.status, duration_seconds=EXCLUDED.duration_seconds,
             hop_seconds=EXCLUDED.hop_seconds, features=EXCLUDED.features, created_at=now()""",
        (track_id, revision, duration, hop_seconds),
    )


def save_visual_features(
    db: Connection,
    track_id: uuid.UUID,
    revision: str,
    duration: float,
    hop_seconds: float,
    features: dict[str, Any],
) -> None:
    """Store the song's visual features."""
    db.execute(
        """INSERT INTO track_visual_features (track_id, revision, status, duration_seconds, hop_seconds, features)
           VALUES (%s, %s, 'complete', %s, %s, %s)
           ON CONFLICT (track_id) DO UPDATE SET revision=EXCLUDED.revision,
             status=EXCLUDED.status, duration_seconds=EXCLUDED.duration_seconds,
             hop_seconds=EXCLUDED.hop_seconds, features=EXCLUDED.features, created_at=now()""",
        (track_id, revision, duration, hop_seconds, Jsonb(features)),
    )


# Chromaprint fingerprints and the recording groups they form.


def has_fingerprint(db: Connection, track_id: uuid.UUID) -> bool:
    """Whether the song has a fingerprint."""
    return (
        fetch_one(db, "SELECT 1 FROM track_fingerprints WHERE track_id=%s", (track_id,)) is not None
    )


def unfingerprinted_duration(db: Connection, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The song's ``duration_seconds`` if it has no fingerprint yet, else ``None``."""
    return fetch_one(
        db,
        """SELECT duration_seconds FROM tracks WHERE id=%s
           AND NOT EXISTS (SELECT 1 FROM track_fingerprints WHERE track_id=%s)""",
        (track_id, track_id),
    )


def save_fingerprint(
    db: Connection, track_id: uuid.UUID, version: str, fingerprint: bytes, duration: float
) -> None:
    """Store the song's Chromaprint fingerprint."""
    db.execute(
        """INSERT INTO track_fingerprints
             (track_id, algorithm, algorithm_version, fingerprint, duration_seconds)
           VALUES (%s, 'chromaprint', %s, %s, %s)
           ON CONFLICT (track_id) DO UPDATE SET fingerprint=EXCLUDED.fingerprint,
             algorithm_version=EXCLUDED.algorithm_version, duration_seconds=EXCLUDED.duration_seconds""",
        (track_id, version, fingerprint, duration),
    )


def fingerprint_candidates(
    db: Connection, track_id: uuid.UUID, shortest: float, longest: float
) -> list[dict[str, Any]]:
    """Other songs' fingerprints (``track_id``, ``fingerprint``, ``duration_seconds``) by length."""
    return fetch_all(
        db,
        """SELECT track_id, fingerprint, duration_seconds FROM track_fingerprints
           WHERE track_id<>%s AND duration_seconds BETWEEN %s AND %s""",
        (track_id, shortest, longest),
    )


def embedding_similarity(
    db: Connection, left: uuid.UUID, right: uuid.UUID, model: str
) -> float | None:
    """Cosine similarity of two songs' newest whole-song audio embeddings of the model."""
    row = fetch_one(
        db,
        """
        WITH latest AS (
          SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding
          FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
          WHERE e.track_id=ANY(%s) AND e.embedding_type='audio-track' AND ar.model_name=%s
          ORDER BY e.track_id, ar.created_at DESC
        )
        SELECT 1 - (a.embedding <=> b.embedding) AS similarity
        FROM latest a JOIN latest b ON a.track_id=%s AND b.track_id=%s
        """,
        ([left, right], model, left, right),
    )
    return float(row["similarity"]) if row and row["similarity"] is not None else None


def save_match_evidence(
    db: Connection,
    left: uuid.UUID,
    right: uuid.UUID,
    score: float,
    duration_delta: float,
    semantic: float | None,
    acoustic: float | None,
    matcher_revision: int,
    evidence: dict[str, Any],
) -> uuid.UUID:
    """Store that two songs (in id order) are one recording; returns the evidence id."""
    row = fetch_one(
        db,
        """INSERT INTO recording_match_evidence
             (left_track_id, right_track_id, decision, chromaprint_score,
              duration_delta_seconds, semantic_similarity, acoustic_similarity,
              matcher_revision, evidence)
           VALUES (%s,%s,'matched',%s,%s,%s,%s,%s,%s)
           ON CONFLICT (left_track_id, right_track_id, matcher_revision)
           DO UPDATE SET decision='matched', chromaprint_score=EXCLUDED.chromaprint_score,
             duration_delta_seconds=EXCLUDED.duration_delta_seconds,
             semantic_similarity=EXCLUDED.semantic_similarity,
             acoustic_similarity=EXCLUDED.acoustic_similarity
           RETURNING id""",
        (left, right, score, duration_delta, semantic, acoustic, matcher_revision, Jsonb(evidence)),
    )
    return cast(dict, row)["id"]


def group_of(db: Connection, track_id: uuid.UUID) -> uuid.UUID | None:
    """The recording group the song belongs to."""
    row = fetch_one(
        db, "SELECT group_id FROM recording_group_members WHERE track_id=%s", (track_id,)
    )
    return row["group_id"] if row else None


def add_recording_group(db: Connection, canonical_track_id: uuid.UUID) -> uuid.UUID:
    """Store a recording group with its canonical song (not yet a member); returns its id."""
    row = fetch_one(
        db,
        "INSERT INTO recording_groups (canonical_track_id) VALUES (%s) RETURNING id",
        (canonical_track_id,),
    )
    return cast(dict, row)["id"]


def add_group_member(
    db: Connection, group_id: uuid.UUID, track_id: uuid.UUID, confidence: float
) -> None:
    """Put the song in the group unless it is in one already."""
    db.execute(
        """INSERT INTO recording_group_members (group_id, track_id, confidence)
           VALUES (%s,%s,%s) ON CONFLICT (track_id) DO NOTHING""",
        (group_id, track_id, confidence),
    )
