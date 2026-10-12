from __future__ import annotations

from .settings import get_settings
from .features import feature_enabled

from dataclasses import dataclass
import uuid
from typing import Iterable

import psycopg

from .melody_config import MELODY_CONTOUR_REVISION
from .audio_descriptors import DESCRIPTOR_REVISION
from .waveforms import WAVEFORM_REVISION
from .visual_features import VISUAL_FEATURE_REVISION


@dataclass(frozen=True)
class ProcessingPlan:
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


def _id_filter(external_ids: Iterable[str] | None) -> tuple[str, list[object]]:
    if external_ids is None:
        return "", []
    values = list(external_ids)
    return " AND ts.external_id=ANY(%s)", [values]


def resolve_library_id(connection: psycopg.Connection, url: str) -> uuid.UUID:
    """Resolve an existing library, never falling back to unscoped processing."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id FROM libraries WHERE lower(rtrim(root_path, '/'))=lower(rtrim(%s, '/'))",
            (url,),
        )
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise ValueError("Expected exactly one library for the source URL")
    return rows[0][0]


def plan_lyrics(
    connection: psycopg.Connection,
    external_ids: Iterable[str] | None = None,
    library_id: uuid.UUID | None = None,
) -> ProcessingPlan:
    restriction, parameters = _id_filter(external_ids)
    if library_id is not None:
        restriction += " AND ts.library_id=%s"
        parameters.append(library_id)
    revision = get_settings().lyrics_revision
    with connection.cursor() as cursor:
        cursor.execute(
            f"""SELECT DISTINCT ts.external_id
                FROM track_sources ts LEFT JOIN lyrics l ON l.track_id=ts.track_id
                WHERE ts.source_type='subsonic'{restriction}
                  AND (l.track_id IS NULL OR l.text IS NULL OR NOT EXISTS (
                    SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                    WHERE e.track_id=ts.track_id AND e.embedding_type='lyrics'
                      AND e.window_index IS NULL AND ar.model_name='bge_m3'
                      AND ar.model_revision=%s
                  )) ORDER BY ts.external_id""",
            [*parameters, revision],
        )
        ids = tuple(str(row[0]) for row in cursor.fetchall())
    return ProcessingPlan(lyrics_external_ids=ids)


def plan_karaoke(
    connection: psycopg.Connection,
    pipeline_revision: str,
    external_ids: Iterable[str] | None = None,
    model_revision: str | None = None,
    library_id: uuid.UUID | None = None,
) -> ProcessingPlan:
    restriction, parameters = _id_filter(external_ids)
    if library_id is not None:
        restriction += " AND ts.library_id=%s"
        parameters.append(library_id)
    with connection.cursor() as cursor:
        if not feature_enabled(connection, "karaoke"):
            return ProcessingPlan()
        cursor.execute(
            f"""SELECT DISTINCT ts.external_id
                FROM track_sources ts JOIN lyrics l ON l.track_id=ts.track_id
                WHERE ts.source_type='subsonic'{restriction}
                  AND NULLIF(btrim(l.text), '') IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM karaoke_lyrics_variants kv
                    WHERE kv.track_id=l.track_id AND kv.bounded=false
                      AND kv.provenance->>'pipeline_revision'=%s
                      AND (%s::text IS NULL OR kv.model_revision=%s::text)
                  ) ORDER BY ts.external_id""",
            [*parameters, pipeline_revision, model_revision, model_revision],
        )
        ids = tuple(str(row[0]) for row in cursor.fetchall())
    return ProcessingPlan(karaoke_external_ids=ids)


def plan_audio(
    connection: psycopg.Connection,
    library_id,
    external_ids: Iterable[str],
    *,
    resolved_track_ids: dict | None = None,
) -> AudioProcessingPlan:
    ids = list(external_ids)
    # Execution can pin identities resolved from actual downloaded bytes. Do not
    # let a concurrent source remap change the artifact plan for those bytes.
    if resolved_track_ids is None:
        source_join = """LEFT JOIN track_sources ts ON ts.library_id=%s AND ts.source_type='subsonic'
                        AND ts.external_id=requested.external_id"""
        source_parameters = [library_id]
    else:
        source_join = """LEFT JOIN unnest(%s::text[],%s::uuid[]) AS ts(external_id,track_id)
                        ON ts.external_id=requested.external_id"""
        source_parameters = [ids, [resolved_track_ids[item] for item in ids]]
    muq_revision = get_settings().muq_revision
    mert_revision = get_settings().mert_revision
    with connection.cursor() as cursor:
        hum_enabled = feature_enabled(connection, "hum")
        cursor.execute(
            f"""SELECT requested.external_id,
                      ts.track_id,
                      EXISTS (SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                              WHERE e.track_id=ts.track_id AND e.embedding_type='audio-track'
                                AND e.window_index IS NULL AND ar.model_name='muq_mulan'
                                AND ar.model_revision=%s
                                AND ar.config->>'coverage'='full-track'
                                AND ar.config->>'window_seconds'='10'
                                AND ar.config->>'stride_seconds'='5'
                                AND ar.config->>'store_window_embeddings'='true') AS has_muq,
                      EXISTS (SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                              WHERE e.track_id=ts.track_id AND e.embedding_type='audio-track'
                                AND e.window_index IS NULL AND ar.model_name='mert'
                                AND ar.model_revision=%s
                                AND ar.config->>'coverage'='full-track'
                                AND ar.config->>'window_seconds'='10'
                                AND ar.config->>'stride_seconds'='5'
                                AND ar.config->>'store_window_embeddings'='true') AS has_mert,
                      EXISTS (SELECT 1 FROM track_fingerprints tf WHERE tf.track_id=ts.track_id) AS has_fingerprint,
                      EXISTS (SELECT 1 FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
                              WHERE mc.track_id=ts.track_id AND ar.model_name='melody_contour'
                                AND ar.model_revision=%s) AS has_melody,
                      EXISTS (SELECT 1 FROM track_audio_descriptors ad
                              WHERE ad.track_id=ts.track_id AND ad.revision=%s
                                AND ad.status='complete') AS has_descriptors,
                      EXISTS (SELECT 1 FROM track_waveforms w
                              WHERE w.track_id=ts.track_id AND w.revision=%s) AS has_waveform,
                      EXISTS (SELECT 1 FROM track_visual_features vf
                              WHERE vf.track_id=ts.track_id AND vf.revision=%s
                                AND vf.status IN ('complete', 'unsupported')) AS has_visual_features
               FROM unnest(%s::text[]) requested(external_id)
               {source_join}""",
            (
                muq_revision,
                mert_revision,
                MELODY_CONTOUR_REVISION,
                DESCRIPTOR_REVISION,
                WAVEFORM_REVISION,
                VISUAL_FEATURE_REVISION,
                ids,
                *source_parameters,
            ),
        )
        rows = cursor.fetchall()
        from .recording_encoder import config_from_env

        recording_config = config_from_env()
        recording_ids = frozenset()
        if recording_config is not None:
            cursor.execute(
                f"""SELECT requested.external_id FROM unnest(%s::text[]) requested(external_id)
                {source_join}
                WHERE NOT EXISTS (SELECT 1 FROM recording_fingerprints f
                    WHERE f.track_id=ts.track_id AND f.representation_id=%s)""",
                (ids, *source_parameters, recording_config.representation_id),
            )
            recording_ids = frozenset(str(row[0]) for row in cursor.fetchall())
    return AudioProcessingPlan(
        frozenset(str(row[0]) for row in rows if not row[2]),
        frozenset(str(row[0]) for row in rows if not row[3]),
        frozenset(str(row[0]) for row in rows if not row[4]),
        frozenset(str(row[0]) for row in rows if hum_enabled and not row[5]),
        frozenset(str(row[0]) for row in rows if not row[6]),
        frozenset(str(row[0]) for row in rows if not row[7]),
        frozenset(str(row[0]) for row in rows if not row[8]),
        recording_ids,
    )
