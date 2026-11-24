"""Curation recipes and the ranked preview of the songs a recipe selects."""

from __future__ import annotations

from __future__ import annotations
from core.errors import ConflictError, IntegrationError, InvalidQueryError, NotFoundError
from core.library.connections import credential_cipher
from datetime import datetime, timedelta, timezone
import secrets
import uuid
import numpy as np
import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from core.curation.concepts import expand_tag_groups
from core.curation.ranking import (
    CURATION_SCORING_REVISION,
    EXAMPLE_COMPONENT_WEIGHTS,
    MATCH_PERCENTILE,
    rank_curation,
)
from core.db.session import session_scope
from core.settings import get_settings
from core.db.models import UserPreference
from core.curation.journeys import (
    normalize_rows as normalize_journey_rows,
    select_multistop_journey,
)
from core.curation.listening import recent_listens, track_listen_counts
from core.lyrics.language import LANGUAGE_NAMES, PRIMARY_SHARE, language_affinity
from core.embeddings.text import shared_lyrics_model
from core.revisions import LYRIC_THEMES_REVISION


class SoundProfile(BaseModel):
    """Library-relative targets for measured audio descriptors."""

    pace: float | None = Field(default=None, ge=0, le=1)
    energy: float | None = Field(default=None, ge=0, le=1)
    brightness: float | None = Field(default=None, ge=0, le=1)
    motion: float | None = Field(default=None, ge=0, le=1)
    vocals: float | None = Field(default=None, ge=0, le=1)
    dynamics: float | None = Field(default=None, ge=0, le=1)
    instrumental_only: bool = False


class CurationRecipe(BaseModel):
    curation_type: str = Field(
        default="combined", pattern="^(combined|language|examples|time_of_day|sonic_journey)$"
    )
    positive_prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    sound_prompts: list[str] = Field(default_factory=list, max_length=12)
    themes_prompts: list[str] = Field(default_factory=list, max_length=12)
    sound_negative_prompts: list[str] = Field(default_factory=list, max_length=12)
    themes_negative_prompts: list[str] = Field(default_factory=list, max_length=12)
    sound_weight: int = Field(default=50, ge=0, le=100)
    sound_profile: SoundProfile = Field(default_factory=SoundProfile)
    positive_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    negative_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    familiarity_percent: int = Field(default=70, ge=0, le=100)
    period_start: str | None = Field(default=None, pattern="^(?:[01]\\d|2[0-3]):[0-5]\\d$")
    period_end: str | None = Field(default=None, pattern="^(?:[01]\\d|2[0-3]):[0-5]\\d$")
    time_of_day_enabled: bool = False
    lookback_days: int = Field(default=7, ge=1, le=90)
    track_limit: int = Field(default=30, ge=5, le=200)
    refresh_mode: str = Field(default="stable", pattern="^(stable|fresh)$")
    target_language: str = Field(default="", pattern="^$|^[a-z]{2,3}$")
    language_strictness: str = Field(default="primarily", pattern="^(only|primarily|sprinkle)$")
    existing_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=200)
    # Sonic journey: ordered waypoints through fused audio+lyrics space.
    journey_start_track_id: uuid.UUID | None = None
    journey_stop_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=5)
    journey_end_track_id: uuid.UUID | None = None
    journey_lyrics_weight: int = Field(default=25, ge=0, le=100)


def curation_corpus(
    user_id: uuid.UUID,
    connection_id: uuid.UUID,
) -> tuple[
    list[dict[str, object]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray] | None],
    np.ndarray | None,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray] | None],
]:
    with (
        psycopg.connect(get_settings().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """WITH selected_embeddings AS (
                 SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding,
                        e.run_id
                 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                 WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
                   AND ar.model_name='muq_mulan'
                 ORDER BY e.track_id, ar.created_at DESC
               )
               SELECT DISTINCT ON (t.id) t.id, t.title, t.artist, t.album, t.duration_seconds,
                      selected_embeddings.embedding, lyrics_embedding.embedding AS lyrics_embedding,
                      selected_embeddings.run_id AS muq_run_id,
                      acoustic_embedding.run_id AS mert_run_id,
                      lyrics_embedding.run_id AS lyrics_run_id,
                      voice_embedding.embedding AS voice_embedding,
                      theme_embedding.embedding AS theme_embedding,
                      acoustic_embedding.embedding AS acoustic_embedding,
                      muq_profile.id AS muq_profile_id, mert_profile.id AS mert_profile_id,
                      ts.external_id AS source_id, ts.source_data->>'coverArt' AS cover_art,
                      member.group_id::text AS recording_group_id,
                      descriptors.descriptors AS audio_descriptors,
                      vocal_activity.activity AS vocal_activity
               FROM selected_embeddings
               JOIN tracks t ON t.id=selected_embeddings.track_id
               JOIN track_sources ts ON ts.track_id=t.id AND ts.source_type='subsonic'
               JOIN libraries l ON l.id=ts.library_id
               JOIN user_track_links ul ON ul.library_id=l.id AND ul.track_id=t.id AND ul.user_id=%s
               JOIN navidrome_connections nc ON nc.id=%s
                 AND lower(rtrim(nc.url, '/'))=lower(rtrim(l.root_path, '/'))
               LEFT JOIN recording_group_members member ON member.track_id=t.id
               LEFT JOIN LATERAL (
                 SELECT descriptors FROM track_audio_descriptors
                 WHERE track_id=t.id AND status IN ('complete', 'partial')
                 ORDER BY created_at DESC LIMIT 1
               ) descriptors ON true
               LEFT JOIN LATERAL (
                 SELECT activity.activity FROM track_vocal_activity activity
                 JOIN current_analysis_runs ar ON ar.id=activity.run_id
                 WHERE activity.track_id=t.id AND ar.kind='voice_classification'
                 ORDER BY activity.created_at DESC LIMIT 1
               ) vocal_activity ON true
               LEFT JOIN LATERAL (
                 SELECT e.embedding::text AS embedding, e.run_id
                 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                 WHERE e.track_id=t.id AND e.embedding_type='audio-track'
                   AND e.window_index IS NULL AND ar.model_name='mert'
                 ORDER BY ar.created_at DESC LIMIT 1
               ) acoustic_embedding ON true
               LEFT JOIN LATERAL (
                 SELECT tap.id FROM current_audio_profiles tap
                 JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                 WHERE tap.track_id=t.id AND tap.source_run_id=selected_embeddings.run_id
                   AND tap.model_name='muq_mulan'
                 ORDER BY tap.created_at DESC LIMIT 1
               ) muq_profile ON true
               LEFT JOIN LATERAL (
                 SELECT tap.id FROM current_audio_profiles tap
                 JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                 WHERE tap.track_id=t.id AND tap.source_run_id=acoustic_embedding.run_id
                   AND tap.model_name='mert'
                 ORDER BY tap.created_at DESC LIMIT 1
               ) mert_profile ON true
               LEFT JOIN LATERAL (
                 SELECT e.embedding::text AS embedding, e.run_id
                 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                 WHERE e.track_id=t.id AND e.embedding_type='lyrics' AND e.window_index IS NULL
                   AND ar.model_name='bge_m3'
                 ORDER BY ar.created_at DESC LIMIT 1
               ) lyrics_embedding ON true
               LEFT JOIN LATERAL (
                 SELECT e.embedding::text AS embedding
                 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                 WHERE e.track_id=t.id AND e.embedding_type='voice-gender' AND ar.model_name='mtg-jamendo-voice-gender-v2'
                 ORDER BY ar.created_at DESC LIMIT 1
               ) voice_embedding ON true
               LEFT JOIN LATERAL (
                 SELECT th.embedding::text AS embedding
                 FROM track_lyric_themes th JOIN lyrics ly ON ly.track_id=th.track_id
                 WHERE th.track_id=t.id AND th.status='ready' AND th.prompt_revision=%s
                   AND th.source_checksum=encode(sha256(convert_to(ly.text, 'UTF8')), 'hex')
                 LIMIT 1
               ) theme_embedding ON true
               ORDER BY t.id, ts.id""",
            (user_id, connection_id, LYRIC_THEMES_REVISION),
        )
        rows = cursor.fetchall()
        profile_ids = [
            value
            for row in rows
            for value in (row.get("muq_profile_id"), row.get("mert_profile_id"))
            if value is not None
        ]
        mode_rows: list[dict[str, object]] = []
        if profile_ids:
            cursor.execute(
                """SELECT profile_id, mode_index, duration_weight, embedding::text AS embedding
                   FROM audio_modes WHERE profile_id=ANY(%s)
                   ORDER BY profile_id, mode_index""",
                (profile_ids,),
            )
            mode_rows = cursor.fetchall()
    if not rows:
        raise ConflictError("No semantic embeddings are available in this library")
    matrix = np.stack([np.fromstring(row.pop("embedding").strip("[]"), sep=",") for row in rows])
    lyrics_available = np.asarray(
        [row.get("lyrics_embedding") is not None for row in rows], dtype=bool
    )
    lyrics_matrix = np.zeros((len(rows), 1024), dtype=np.float32)
    for index, row in enumerate(rows):
        encoded = row.pop("lyrics_embedding")
        if encoded is not None:
            lyrics_matrix[index] = np.fromstring(str(encoded).strip("[]"), sep=",")
    theme_available = np.asarray(
        [row.get("theme_embedding") is not None for row in rows], dtype=bool
    )
    theme_matrix = np.zeros((len(rows), 1024), dtype=np.float32)
    for index, row in enumerate(rows):
        encoded = row.pop("theme_embedding")
        if encoded is not None:
            theme_matrix[index] = np.fromstring(str(encoded).strip("[]"), sep=",")
    voice_available = np.asarray(
        [row.get("voice_embedding") is not None for row in rows], dtype=bool
    )
    voice_matrix = np.zeros((len(rows), 3), dtype=np.float32)
    for index, row in enumerate(rows):
        encoded = row.pop("voice_embedding")
        if encoded is not None:
            voice_matrix[index] = np.fromstring(str(encoded).strip("[]"), sep=",")
    acoustic_available = np.asarray(
        [row.get("acoustic_embedding") is not None for row in rows], dtype=bool
    )
    acoustic_dimension = next(
        (
            len(np.fromstring(str(row["acoustic_embedding"]).strip("[]"), sep=","))
            for row in rows
            if row.get("acoustic_embedding") is not None
        ),
        0,
    )
    acoustic_matrix = np.zeros((len(rows), acoustic_dimension), dtype=np.float32)
    for index, row in enumerate(rows):
        encoded = row.pop("acoustic_embedding")
        if encoded is not None:
            acoustic_matrix[index] = np.fromstring(str(encoded).strip("[]"), sep=",")
    grouped_modes: dict[uuid.UUID, list[dict[str, object]]] = {}
    for mode_row in mode_rows:
        grouped_modes.setdefault(mode_row["profile_id"], []).append(mode_row)
    modes_by_profile: dict[uuid.UUID, tuple[np.ndarray, np.ndarray]] = {}
    for profile_id, members in grouped_modes.items():
        if members:
            modes_by_profile[profile_id] = (
                np.stack(
                    [np.fromstring(str(row["embedding"]).strip("[]"), sep=",") for row in members]
                ),
                np.asarray([row["duration_weight"] for row in members], dtype=np.float32),
            )
    for row in rows:
        row["representation_runs"] = {
            model: str(run_id)
            for model, run_id in (
                ("muq_mulan", row.pop("muq_run_id")),
                ("mert", row.pop("mert_run_id")),
                ("bge_m3", row.pop("lyrics_run_id")),
            )
            if run_id is not None
        }
    semantic_modes = [modes_by_profile.get(row.pop("muq_profile_id")) for row in rows]
    acoustic_modes = [modes_by_profile.get(row.pop("mert_profile_id")) for row in rows]
    for row in rows:
        descriptors = row.pop("audio_descriptors", None) or {}
        rhythm = descriptors.get("rhythm") or {}
        spectral = descriptors.get("spectral_curve") or []
        loudness = descriptors.get("loudness") or {}
        activity = row.pop("vocal_activity", None) or {}
        activity_windows = activity.get("windows") or []

        def _mean(values: list[object], field: str) -> float | None:
            usable = [
                float(item[field])
                for item in values
                if isinstance(item, dict)
                and item.get(field) is not None
                and np.isfinite(float(item[field]))
            ]
            return float(np.mean(usable)) if usable else None

        row["sound_descriptors"] = {
            "pace": rhythm.get("bpm"),
            "energy": descriptors.get("rms_dbfs"),
            "brightness": _mean(spectral, "centroid_hz"),
            "motion": _mean(spectral, "positive_spectral_flux"),
            "vocals": _mean(activity_windows, "vocal_activation"),
            "dynamics": loudness.get("range_lu"),
        }
    return (
        rows,
        matrix,
        lyrics_matrix,
        lyrics_available,
        voice_matrix,
        voice_available,
        semantic_modes,
        acoustic_matrix if acoustic_dimension else None,
        acoustic_available,
        acoustic_modes,
        theme_matrix,
        theme_available,
    )


# Use theme descriptions once they cover this share of the tracks that have lyrics.
THEME_COVERAGE_THRESHOLD = 0.5


def theme_overlay(
    lyrics_matrix: np.ndarray,
    lyrics_available: np.ndarray,
    theme_matrix: np.ndarray,
    theme_available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The vectors the Themes channel compares queries and examples against.

    Theme descriptions (lyric_themes.py) replace whole-lyric embeddings once they cover most
    of the tracks with lyrics. The two live in different regions of the BGE-M3 space, so they
    are never ranked together: a track with lyrics but no description yet is treated like one
    without lyrics (neutral) until a sync describes it.
    """
    lyric_count = int(lyrics_available.sum())
    covered = int((theme_available & lyrics_available).sum())
    if lyric_count == 0 or covered < THEME_COVERAGE_THRESHOLD * lyric_count:
        return lyrics_matrix, lyrics_available, False
    return theme_matrix, theme_available.copy(), True


SONIC_AUDIO_SPLIT = (0.70, 0.30)  # MuQ : MERT within the audio share


SONIC_SCORING_REVISION = 1


def preview_sonic_journey(
    request: CurationRecipe,
    rows: list[dict[str, object]],
    matrix: np.ndarray,
    acoustic_matrix: np.ndarray | None,
    lyrics_matrix: np.ndarray,
    lyrics_available: np.ndarray,
) -> dict[str, object]:
    identifiers = {str(row["id"]): index for index, row in enumerate(rows)}
    waypoints = [
        request.journey_start_track_id,
        *request.journey_stop_track_ids,
        request.journey_end_track_id,
    ]
    if not waypoints[0] or not waypoints[-1]:
        raise InvalidQueryError("Pick a start and an end track")
    if waypoints[0] == waypoints[-1]:
        raise InvalidQueryError("Journey start and end must be different tracks")
    indices: list[int] = []
    for waypoint in waypoints:
        index = identifiers.get(str(waypoint))
        if index is None:
            raise NotFoundError("A journey stop lacks audio analysis")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise InvalidQueryError("Each journey stop must be a different track")
    if acoustic_matrix is None:
        raise ConflictError("MERT analysis has not run for this library yet")

    def _norm(values: np.ndarray) -> np.ndarray:
        return normalize_journey_rows(np.asarray(values, dtype=np.float32))

    lyrics_share = request.journey_lyrics_weight / 100.0
    audio = np.concatenate(
        [
            _norm(matrix) * np.sqrt(SONIC_AUDIO_SPLIT[0]),
            _norm(acoustic_matrix) * np.sqrt(SONIC_AUDIO_SPLIT[1]),
        ],
        axis=1,
    )
    fused = np.concatenate(
        [
            audio * np.sqrt(1 - lyrics_share),
            _norm(lyrics_matrix) * np.sqrt(lyrics_share),
        ],
        axis=1,
    )

    with (
        psycopg.connect(get_settings().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT track_id::text AS id, group_id::text AS group_id FROM recording_group_members WHERE track_id=ANY(%s)",
            ([row["id"] for row in rows],),
        )
        groups_by_id = {row["id"]: row["group_id"] for row in cursor.fetchall()}
    steps = select_multistop_journey(
        fused,
        indices,
        request.track_limit,
        [row.get("artist") for row in rows],
        [groups_by_id.get(str(row["id"])) for row in rows],
    )
    tracks = []
    for position, (index, target_similarity, transition_similarity) in enumerate(steps):
        row = rows[index]
        tracks.append(
            {
                "id": row["id"],
                "title": row["title"],
                "artist": row.get("artist"),
                "album": row.get("album"),
                "duration_seconds": row.get("duration_seconds"),
                "source_id": row.get("source_id"),
                "cover_art": row.get("cover_art"),
                "position": position,
                "score": float(target_similarity),
                "percentile": float(target_similarity),
                "retained": False,
                "waypoint": index in indices,
                "evidence": {
                    "target_similarity": float(target_similarity),
                    "transition_similarity": float(transition_similarity),
                    "lyrics_available": bool(lyrics_available[index])
                    if index < len(lyrics_available)
                    else False,
                },
            }
        )
    lyrics_weight = request.journey_lyrics_weight / 100.0
    return {
        "tracks": tracks,
        "references": {"positive": [], "negative": []},
        "corpus_size": len(rows),
        "selection": {
            "requested": request.track_limit,
            "selected": len(tracks),
            "shortfall": max(0, request.track_limit - len(tracks)),
            "match_basis": "sonic_journey",
        },
        "curation_type": "sonic_journey",
        "model": "muq_mulan+mert+bge_m3",
        "weights": {
            "muq_mulan": SONIC_AUDIO_SPLIT[0] * (1 - lyrics_weight),
            "mert": SONIC_AUDIO_SPLIT[1] * (1 - lyrics_weight),
            "lyrics": lyrics_weight,
        },
        "journey": {
            "start_track_id": str(request.journey_start_track_id),
            "stop_track_ids": [str(value) for value in request.journey_stop_track_ids],
            "end_track_id": str(request.journey_end_track_id),
            "lyrics_weight": request.journey_lyrics_weight,
            "waypoint_count": len(indices),
        },
        "scoring_revision": SONIC_SCORING_REVISION,
        "lyrics_coverage": int(lyrics_available.sum()),
    }


def preview_curation(
    user_id: uuid.UUID,
    connection_id: uuid.UUID,
    request: CurationRecipe,
) -> dict[str, object]:
    (
        rows,
        matrix,
        lyrics_matrix,
        lyrics_available,
        voice_matrix,
        voice_available,
        semantic_modes,
        acoustic_matrix,
        acoustic_available,
        acoustic_modes,
        theme_matrix,
        theme_available,
    ) = curation_corpus(user_id, connection_id)
    if request.curation_type == "sonic_journey":
        return preview_sonic_journey(
            request, rows, matrix, acoustic_matrix, lyrics_matrix, lyrics_available
        )
    raw_lyrics_coverage = int(lyrics_available.sum())
    lyrics_matrix, lyrics_available, themes_active = theme_overlay(
        lyrics_matrix, lyrics_available, theme_matrix, theme_available
    )
    language_map: dict[str, dict[str, object]] = {}
    if request.target_language:
        with (
            psycopg.connect(get_settings().database_url, row_factory=dict_row) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT l.track_id::text AS id, l.provenance->'languages' AS distribution
                   FROM lyrics l WHERE l.provenance ? 'languages'"""
            )
            language_map = {
                row["id"]: row["distribution"] for row in cursor.fetchall() if row["distribution"]
            }
    time_of_day_enabled = request.time_of_day_enabled or request.curation_type == "time_of_day"
    if time_of_day_enabled and (not request.period_start or not request.period_end):
        raise InvalidQueryError("Choose a start and end time")
    if request.curation_type == "language" and not (
        request.positive_prompt.strip()
        or request.sound_prompts
        or request.themes_prompts
        or request.target_language
    ):
        raise InvalidQueryError(
            "Pick a language or add a positive direction for this language curation",
        )
    if request.curation_type == "examples" and not request.positive_track_ids:
        raise InvalidQueryError("Add at least one Songs like track")
    overlap = set(request.positive_track_ids) & set(request.negative_track_ids)
    if overlap:
        raise InvalidQueryError("A track cannot appear in both Songs like and Songs not like")
    with session_scope() as session:
        preference = session.get(UserPreference, user_id)
        if preference is None:
            raise NotFoundError("User preferences are unavailable")
        timezone_name = preference.timezone
        lastfm_username = preference.lastfm_username
        encrypted_key = (
            bytes(preference.lastfm_api_key_encrypted)
            if preference.lastfm_api_key_encrypted
            else None
        )
    listen_counts: dict[str, int] | None = None
    recent_ids: set[str] | None = None
    period_counts: dict[str, int] = {}
    if lastfm_username and encrypted_key:
        try:
            api_key = credential_cipher().decrypt(encrypted_key).decode()
            listens = recent_listens(
                lastfm_username,
                api_key,
                datetime.now(timezone.utc) - timedelta(days=request.lookback_days),
            )
            all_counts, period_counts = track_listen_counts(
                rows,
                listens,
                timezone_name,
                request.period_start if time_of_day_enabled else None,
                request.period_end if time_of_day_enabled else None,
            )
            listen_counts = period_counts if time_of_day_enabled else all_counts
            recent_ids = set(all_counts)
        except Exception as error:
            raise IntegrationError("Could not read Last.fm listening history") from error
    elif time_of_day_enabled:
        raise ConflictError(
            "Connect Last.fm in Settings before creating a time-of-day curation",
        )
    sound_profile = request.sound_profile.model_dump(exclude_none=True, exclude_defaults=True)
    instrumental_only = bool(sound_profile.pop("instrumental_only", False))
    effective_positive_ids = list(request.positive_track_ids)
    time_of_day_track_ids: list[uuid.UUID] = []
    if time_of_day_enabled:
        time_of_day_track_ids = [uuid.UUID(identifier) for identifier in period_counts]
        if not time_of_day_track_ids:
            raise ConflictError(
                "No matched listens were found in that time period during the lookback window",
            )
    sound_tags = [tag.strip() for tag in request.sound_prompts if tag.strip()]
    theme_tags = [tag.strip() for tag in request.themes_prompts if tag.strip()]
    sound_negatives = [tag.strip() for tag in request.sound_negative_prompts if tag.strip()]
    theme_negatives = [tag.strip() for tag in request.themes_negative_prompts if tag.strip()]
    structured = bool(sound_tags or theme_tags or sound_negatives or theme_negatives)
    semantic_prompts = (
        sound_tags
        if structured
        else ([request.positive_prompt] if request.positive_prompt.strip() else [])
    )
    theme_prompts = (
        theme_tags
        if structured
        else ([request.positive_prompt] if request.positive_prompt.strip() else [])
    )
    visible_ids = {str(row["id"]) for row in rows}
    has_language_constraint = bool(request.target_language)
    if (
        not has_language_constraint
        and not semantic_prompts
        and not theme_prompts
        and not sound_profile
        and not instrumental_only
        and not any(
            str(value) in visible_ids for value in [*effective_positive_ids, *time_of_day_track_ids]
        )
    ):
        raise InvalidQueryError("None of the Songs like tracks are available in this library")
    lyrics_model = shared_lyrics_model()

    def _tag_query_centers(tags: list[str]) -> list[np.ndarray]:
        centers: list[np.ndarray] = []
        for group in expand_tag_groups(tags):
            embedded = lyrics_model.embed_queries(group)
            center = np.asarray(embedded, dtype=np.float32).mean(axis=0)
            centers.append(center / max(float(np.linalg.norm(center)), 1e-8))
        return centers

    positive_queries = _tag_query_centers(theme_prompts) if theme_prompts else None
    lyrics_negatives = (
        theme_negatives
        if structured
        else ([request.negative_prompt] if request.negative_prompt.strip() else [])
    )
    negative_queries = (
        np.asarray(_tag_query_centers(lyrics_negatives), dtype=np.float32)
        if lyrics_negatives
        else None
    )
    shuffle_seed = secrets.randbits(63)
    language_mode = bool(request.target_language)
    language_eligible_ids: set[str] | None = None
    if language_mode and request.language_strictness == "only":
        language_eligible_ids = {
            str(row["id"])
            for row in rows
            if language_affinity(language_map.get(str(row["id"])), request.target_language)
            >= PRIMARY_SHARE
        }
        if not language_eligible_ids:
            raise ConflictError(
                (
                    "No tracks with confident "
                    f"{LANGUAGE_NAMES.get(request.target_language, request.target_language)} "
                    "lyrics were found in this library"
                ),
            )
    rank_limit = (
        len(rows) if language_mode and language_eligible_ids is None else request.track_limit
    )
    expanded_sound_prompts = expand_tag_groups(sound_tags) if structured and sound_tags else None
    expanded_sound_negatives = (
        expand_tag_groups(sound_negatives) if structured and sound_negatives else None
    )
    has_sound_profile = bool(sound_profile)
    tracks, references = rank_curation(
        rows,
        matrix,
        request.positive_prompt,
        request.negative_prompt,
        rank_limit,
        request.refresh_mode,
        [str(value) for value in request.existing_track_ids],
        [str(value) for value in effective_positive_ids],
        [str(value) for value in request.negative_track_ids],
        lyrics_matrix,
        lyrics_available,
        positive_queries,
        negative_queries,
        listen_counts,
        recent_ids,
        request.familiarity_percent,
        shuffle_seed,
        voice_matrix=voice_matrix,
        voice_available=voice_available,
        sound_prompts=expanded_sound_prompts,
        themes_prompts=expand_tag_groups(theme_tags) if structured and theme_tags else None,
        sound_negative_prompts=expanded_sound_negatives,
        themes_negative_prompts=expand_tag_groups(theme_negatives)
        if structured and theme_negatives
        else None,
        sound_weight=request.sound_weight if structured else None,
        sound_profile=sound_profile,
        instrumental_only=instrumental_only,
        semantic_modes=semantic_modes,
        acoustic_matrix=acoustic_matrix,
        acoustic_available=acoustic_available,
        acoustic_modes=acoustic_modes,
        context_track_ids=[str(value) for value in time_of_day_track_ids],
        eligible_track_ids=language_eligible_ids,
        # Profile targets are preference scores, not semantic relevance claims.
        # A profile-only recipe therefore ranks the measured library without
        # applying the embedding match cutoff intended for text and examples.
        minimum_match_percentile=(
            0.0
            if (has_sound_profile or instrumental_only)
            and not (
                semantic_prompts or theme_prompts or effective_positive_ids or time_of_day_track_ids
            )
            else MATCH_PERCENTILE
        ),
    )
    language_report: dict[str, object] | None = None
    if language_mode:
        affinity = {
            str(track["id"]): language_affinity(
                language_map.get(str(track["id"])), request.target_language
            )
            for track in tracks
        }
        if request.language_strictness == "primarily":
            ranked = sorted(tracks, key=lambda track: float(track["percentile"]), reverse=True)
            primary = [track for track in ranked if affinity[str(track["id"])] >= PRIMARY_SHARE]
            filler = [track for track in ranked if affinity[str(track["id"])] < PRIMARY_SHARE]
            tracks = (primary + filler)[: request.track_limit]
            language_report = {
                "target": request.target_language,
                "strictness": "primarily",
                "matched_tracks": len(primary),
                "requested": request.track_limit,
            }
        elif request.language_strictness == "sprinkle":
            boosted = [
                (0.85 * float(track["percentile"]) + 0.15 * affinity[str(track["id"])], track)
                for track in tracks
            ]
            tracks = [track for _, track in sorted(boosted, key=lambda item: -item[0])]
            tracks = tracks[: request.track_limit]
            language_report = {
                "target": request.target_language,
                "strictness": "sprinkle",
                "matched_tracks": sum(value >= PRIMARY_SHARE for value in affinity.values()),
                "requested": request.track_limit,
            }
        else:
            language_report = {
                "target": request.target_language,
                "strictness": "only",
                "matched_tracks": len(tracks),
                "requested": request.track_limit,
            }
    if (
        not structured
        and not request.positive_prompt.strip()
        and not (effective_positive_ids or time_of_day_track_ids)
    ):
        weights = {"semantic": 0.0, "lyrics": 0.0}
    elif structured and (sound_tags or sound_negatives) and not (theme_tags or theme_negatives):
        weights = {"semantic": 1.0, "lyrics": 0.0}
    elif structured and (theme_tags or theme_negatives) and not (sound_tags or sound_negatives):
        weights = {"semantic": 0.0, "lyrics": 1.0}
    elif structured:
        weights = {
            "semantic": request.sound_weight / 100.0,
            "lyrics": (100 - request.sound_weight) / 100.0,
        }
    else:
        weights = {"semantic": 0.45, "lyrics": 0.55}
    signal_weights: dict[str, float] = {}
    has_sound = bool(sound_tags or sound_negatives)
    has_themes = bool(theme_tags or theme_negatives)
    if has_sound and has_themes:
        signal_weights["sound"] = request.sound_weight / 100.0
        signal_weights["themes"] = (100 - request.sound_weight) / 100.0
    elif has_sound:
        signal_weights["sound"] = 1.0
    elif has_themes:
        signal_weights["themes"] = 1.0
    elif request.positive_prompt.strip():
        signal_weights["free_form"] = 1.0
    if effective_positive_ids or request.negative_track_ids:
        signal_weights["examples"] = 1.0
    if time_of_day_track_ids:
        signal_weights["time_of_day"] = 1.0
    if has_sound_profile:
        signal_weights["sound_profile"] = 1.0
    if instrumental_only:
        signal_weights["instrumental_only"] = 1.0
    signal_total = sum(signal_weights.values())
    if signal_total:
        signal_weights = {name: value / signal_total for name, value in signal_weights.items()}
    return {
        "tracks": tracks,
        "references": references,
        "corpus_size": len(rows),
        "selection": {
            "requested": request.track_limit,
            "selected": len(tracks),
            "shortfall": max(0, request.track_limit - len(tracks)),
            "below_threshold_filling": False,
            "match_basis": "library_relative",
            "confidence_calibrated": False,
        },
        "representation_runs": {
            model: sorted(
                {
                    row["representation_runs"][model]
                    for row in rows
                    if model in row["representation_runs"]
                }
            )
            for model in ("muq_mulan", "mert", "bge_m3")
        },
        "curation_type": request.curation_type,
        "model": "muq_mulan+mert+bge_m3",
        "weights": weights,
        "signal_weights": signal_weights,
        "scoring_revision": CURATION_SCORING_REVISION,
        "example_component_weights": EXAMPLE_COMPONENT_WEIGHTS,
        "audio_profile_coverage": {
            "muq_mulan": sum(profile is not None for profile in semantic_modes),
            "mert": sum(profile is not None for profile in acoustic_modes),
            "mert_global": int(acoustic_available.sum()),
        },
        "lyrics_coverage": raw_lyrics_coverage,
        "theme_coverage": int(theme_available.sum()),
        "lyrics_representation": "themes" if themes_active else "lyrics",
        "shuffle_seed": shuffle_seed,
        "language": language_report,
        "familiarity": {
            "percent": request.familiarity_percent,
            "active": listen_counts is not None,
            "lookback_days": request.lookback_days,
            "familiar_tracks": sum(
                track["evidence"]["selection_pool"] == "familiar" for track in tracks
            ),
            "discovery_tracks": sum(
                track["evidence"]["selection_pool"] == "discovery" for track in tracks
            ),
            "matched_listens": sum((listen_counts or {}).values()),
        },
    }
