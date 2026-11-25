"""Scoped discovery over existing embeddings; no plugin-specific analysis index."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import uuid

from fastapi import HTTPException
import numpy as np

from .artists import (
    fit_artist_profile,
    representative_indices,
    soft_chamfer_similarity,
    weighted_center,
)
from core.curation.journeys import normalize_rows, select_journey, spherical_targets
from .navidrome_integration import connect

DIMENSIONS = {"muq": 512, "mert": 768, "lyrics": 1024}


@dataclass
class Corpus:
    rows: list[dict]
    vectors: dict[str, np.ndarray]
    available: dict[str, np.ndarray]

    def seed(self, source_id):
        for index, row in enumerate(self.rows):
            if row["source_id"] == source_id:
                return index
        raise HTTPException(404, "Track is not analyzed or is outside this key's library")


def library_namespace(principal):
    return uuid.uuid5(uuid.NAMESPACE_URL, principal["url"].rstrip("/"))


def load_corpus(principal):
    with connect() as db:
        rows = db.execute(
            """
            SELECT t.id,t.title,t.artist,t.album,t.duration_seconds,ts.external_id AS source_id,
                   coalesce(ts.source_data->>'artistId',ts.source_data#>>'{artists,0,id}') AS artist_id,
                   coalesce(ts.source_data->>'artist',t.artist) AS source_artist,
                   member.group_id::text AS recording_group_id,
                   muq.embedding::text AS muq,mert.embedding::text AS mert,
                   lyric.embedding::text AS lyrics
            FROM user_source_memberships visible
            JOIN libraries lib ON lib.id=visible.library_id AND lib.namespace=%s
            JOIN track_sources ts ON ts.library_id=visible.library_id
                AND ts.external_id=visible.external_id AND ts.source_type='subsonic'
            JOIN tracks t ON t.id=ts.track_id
            LEFT JOIN recording_group_members member ON member.track_id=t.id
            LEFT JOIN LATERAL (
                SELECT e.embedding FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                WHERE e.track_id=t.id AND e.embedding_type='audio-track' AND e.window_index IS NULL
                    AND ar.model_name='muq_mulan' ORDER BY ar.created_at DESC LIMIT 1
            ) muq ON true
            LEFT JOIN LATERAL (
                SELECT e.embedding FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                WHERE e.track_id=t.id AND e.embedding_type='audio-track' AND e.window_index IS NULL
                    AND ar.model_name='mert' ORDER BY ar.created_at DESC LIMIT 1
            ) mert ON true
            LEFT JOIN LATERAL (
                SELECT e.embedding FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                WHERE e.track_id=t.id AND e.embedding_type='lyrics' AND e.window_index IS NULL
                    AND ar.model_name='bge_m3' ORDER BY ar.created_at DESC LIMIT 1
            ) lyric ON true
            WHERE visible.user_id=%s AND (muq.embedding IS NOT NULL OR mert.embedding IS NOT NULL
                OR lyric.embedding IS NOT NULL) ORDER BY t.id,ts.external_id
        """,
            (library_namespace(principal), principal["user_id"]),
        ).fetchall()
    return corpus_from_rows(rows)


def corpus_from_rows(rows):
    vectors, available = {}, {}
    for key, dimension in DIMENSIONS.items():
        matrix = np.zeros((len(rows), dimension), dtype=np.float32)
        present = np.zeros(len(rows), dtype=bool)
        for index, row in enumerate(rows):
            encoded = row.get(key)
            if encoded is None:
                continue
            vector = (
                np.fromstring(encoded.strip("[]"), sep=",")
                if isinstance(encoded, str)
                else np.asarray(encoded)
            )
            if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
                raise HTTPException(
                    503, "An analysis representation is invalid; reanalyze this library"
                )
            if np.linalg.norm(vector) > 1e-8:
                matrix[index] = vector
                present[index] = True
        vectors[key], available[key] = normalize_rows(matrix), present
    return Corpus(rows, vectors, available)


def audio_scores(corpus, seed, profile):
    weights = {
        "muq": profile.musical_semantic_weight / 100,
        "mert": 1 - profile.musical_semantic_weight / 100,
    }
    numerator = np.zeros(len(corpus.rows), dtype=np.float32)
    denominator = np.zeros(len(corpus.rows), dtype=np.float32)
    for key, weight in weights.items():
        mask = corpus.available[key] & corpus.available[key][seed]
        numerator += (corpus.vectors[key] @ corpus.vectors[key][seed]) * mask * weight
        denominator += mask * weight
    return numerator / np.maximum(denominator, 1e-8), denominator > 0


def similarity_scores(corpus, seed, profile):
    audio, has_audio = audio_scores(corpus, seed, profile)
    has_lyrics = corpus.available["lyrics"] & corpus.available["lyrics"][seed]
    lyrics = corpus.vectors["lyrics"] @ corpus.vectors["lyrics"][seed]
    music_weight = profile.musical_weight / 100
    lyric_weight = 1 - music_weight
    aweight = has_audio.astype(np.float32) * music_weight
    lweight = has_lyrics.astype(np.float32) * lyric_weight
    if profile.missing_lyrics == "audio" and lyric_weight > 0:
        # Fallback means audio-only for a pair, never treating absent lyrics as a zero match.
        aweight = np.where(~has_lyrics & has_audio, 1.0, aweight)
    denominator = aweight + lweight
    scores = (audio * aweight + lyrics * lweight) / np.maximum(denominator, 1e-8)
    valid = denominator > 0
    if profile.missing_lyrics == "exclude" and lyric_weight > 0:
        valid &= has_lyrics
    return np.where(valid, np.clip(scores, 0, 1), -np.inf)


def song_ref(row):
    return {
        "id": row["source_id"],
        "name": row["title"],
        "artist": row.get("source_artist") or row.get("artist") or "",
        "album": row.get("album") or "",
        "durationMs": round(float(row["duration_seconds"]) * 1000),
    }


def diverse_indices(corpus, scores, count, profile, *, seed=None, allowed=None):
    seen_tracks, seen_groups, artists = set(), set(), Counter()
    if seed is not None:
        seen_tracks.add(corpus.rows[seed]["id"])
        group = corpus.rows[seed].get("recording_group_id")
        if group:
            seen_groups.add(group)
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        index = int(index)
        row = corpus.rows[index]
        group = row.get("recording_group_id")
        artist = row.get("artist_id") or (row.get("source_artist") or "").casefold()
        if not np.isfinite(scores[index]) or (allowed is not None and index not in allowed):
            continue
        if row["id"] in seen_tracks or (group and group in seen_groups):
            continue
        if artists[artist] >= profile.max_per_artist:
            continue
        selected.append(index)
        seen_tracks.add(row["id"])
        if group:
            seen_groups.add(group)
        artists[artist] += 1
        if len(selected) >= count:
            break
    return selected


def similar_tracks(corpus, source_id, count, profile):
    seed = corpus.seed(source_id)
    scores = similarity_scores(corpus, seed, profile)
    if not np.isfinite(scores[seed]):
        raise HTTPException(409, "Seed track lacks the analysis required by your ranking settings")
    return {
        "matches": [
            {"song": song_ref(corpus.rows[index]), "similarity": float(scores[index])}
            for index in diverse_indices(corpus, scores, count, profile, seed=seed)
        ]
    }


def blended_matrix(corpus, profile):
    audio = normalize_rows(
        np.concatenate(
            [
                corpus.vectors["muq"] * np.sqrt(profile.musical_semantic_weight / 100),
                corpus.vectors["mert"] * np.sqrt(1 - profile.musical_semantic_weight / 100),
            ],
            axis=1,
        )
    )
    audio_weight = np.full(len(corpus.rows), profile.musical_weight / 100)
    lyric_weight = np.full(len(corpus.rows), 1 - profile.musical_weight / 100)
    if profile.missing_lyrics == "audio":
        audio_weight[~corpus.available["lyrics"]] = 1
        lyric_weight[~corpus.available["lyrics"]] = 0
    matrix = normalize_rows(
        np.concatenate(
            [
                audio * np.sqrt(audio_weight[:, None]),
                corpus.vectors["lyrics"] * np.sqrt(lyric_weight[:, None]),
            ],
            axis=1,
        )
    )
    valid = np.linalg.norm(matrix, axis=1) > 1e-8
    if profile.missing_lyrics == "exclude" and profile.musical_weight < 100:
        valid &= corpus.available["lyrics"]
    return matrix, valid


def sonic_path(corpus, start_id, end_id, count, profile):
    if count < 2:
        raise HTTPException(400, "A sonic path requires room for both endpoints")
    start, end = corpus.seed(start_id), corpus.seed(end_id)
    if corpus.rows[start]["id"] == corpus.rows[end]["id"]:
        raise HTTPException(400, "Path endpoints must be different recordings")
    start_group = corpus.rows[start].get("recording_group_id")
    if start_group and start_group == corpus.rows[end].get("recording_group_id"):
        raise HTTPException(400, "Path endpoints must be different recordings")
    # When an endpoint lacks lyrics, use one consistent audio space for the entire path.
    effective = profile
    if profile.missing_lyrics == "audio" and not (
        corpus.available["lyrics"][start] and corpus.available["lyrics"][end]
    ):
        effective = profile.model_copy(update={"musical_weight": 100})
    matrix, valid = blended_matrix(corpus, effective)
    if not valid[start] or not valid[end]:
        raise HTTPException(
            409, "Path endpoints lack the analysis required by your ranking settings"
        )
    # Collapse alternate source IDs for a recording before interpolating.
    chosen = [start, end]
    seen = {corpus.rows[start]["id"], corpus.rows[end]["id"]}
    for index, row in enumerate(corpus.rows):
        if valid[index] and row["id"] not in seen:
            chosen.append(index)
            seen.add(row["id"])
    rows = [corpus.rows[index] for index in chosen]
    values = matrix[chosen]
    targets = spherical_targets(values[0], values[1], min(count, len(rows)))
    steps = select_journey(
        values,
        targets,
        0,
        1,
        [row.get("artist_id") or row.get("source_artist") for row in rows],
        [row.get("recording_group_id") for row in rows],
        max_per_artist=profile.max_per_artist,
    )
    return {
        "matches": [{"song": song_ref(rows[index]), "similarity": -1.0} for index, _, _ in steps]
    }


def artist_groups(corpus, profile):
    matrix, valid = blended_matrix(corpus, profile)
    groups, seen = {}, set()
    for index, row in enumerate(corpus.rows):
        key = row.get("artist_id")
        identity = (key, row["id"])
        if key and valid[index] and identity not in seen:
            groups.setdefault(key, []).append(index)
            seen.add(identity)
    return matrix, groups


def similar_artist_profiles(corpus, artist_id, count, profile):
    matrix, groups = artist_groups(corpus, profile)
    if artist_id not in groups:
        raise HTTPException(404, "Artist is not analyzed or is outside this key's library")
    target = fit_artist_profile(matrix[groups[artist_id]])
    center = weighted_center(target)
    candidates = []
    for key, indices in groups.items():
        if key != artist_id:
            candidates.append(
                (key, float(normalize_rows(matrix[indices].mean(axis=0)[None, :])[0] @ center))
            )
    ranked = []
    for key, _ in sorted(candidates, key=lambda item: item[1], reverse=True)[: max(count * 3, 30)]:
        other = fit_artist_profile(matrix[groups[key]])
        similarity, _, _, _ = soft_chamfer_similarity(target, other)
        ranked.append((key, max(0.0, min(1.0, similarity)), other))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return matrix, groups, target, ranked[:count]


def similar_artists(corpus, artist_id, count, profile):
    _, groups, _, ranked = similar_artist_profiles(corpus, artist_id, count, profile)
    return {
        "artists": [
            {"id": key, "name": corpus.rows[groups[key][0]]["source_artist"]}
            for key, _, _ in ranked
        ]
    }


def artist_radio(corpus, artist_id, count, profile):
    matrix, groups, target, ranked = similar_artist_profiles(
        corpus, artist_id, min(count, 20), profile
    )
    target_indices = groups[artist_id]
    # Mix representative seed-artist songs with the strongest related artist facets.
    representative = representative_indices(matrix[target_indices], target)
    scores = np.full(len(corpus.rows), -np.inf)
    for order, local in enumerate(representative):
        scores[target_indices[local]] = 1 - order * 0.001
    for key, score, other in ranked:
        for order, local in enumerate(representative_indices(matrix[groups[key]], other)):
            scores[groups[key][local]] = score - order * 0.001
    selected = diverse_indices(corpus, scores, count, profile)
    return {"songs": [song_ref(corpus.rows[index]) for index in selected]}
