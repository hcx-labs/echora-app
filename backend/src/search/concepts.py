"""Concept lens: how strongly each of the user's songs matches each of a few named concepts.

Concepts are predefined or the user's own (prompts plus example songs). Scores are calibrated
as percentiles within the user's library, over MuQ-MuLan audio, BGE-M3 lyrics or both
(``hybrid``). The corpora are persistent (``backend.search.corpora``); the lens itself is a
per-request search on the bounded pool, since it embeds the concepts' prompts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from backend.search.corpora import lyrics_corpus, semantic_corpus
from backend.search.service import SearchService
from core.curation import store as curation
from core.curation.concepts import (
    combine_concept_percentiles,
    empirical_percentiles,
    predefined_concepts,
    score_concept,
)
from core.db.connection import connect
from core.embeddings.text import shared_lyrics_model
from core.errors import InvalidQueryError, NotFoundError


@dataclass(frozen=True)
class LensQuery:
    """Concept names, the percentile a song must reach and the representation."""

    concepts: list[str]
    minimum_percentile: float
    representation: str


class ConceptLens(SearchService[LensQuery, dict]):
    """Scores of one user's songs against named concepts."""

    name = "concept-lens"

    scope: uuid.UUID

    def search(self, query: LensQuery) -> dict:
        return concept_lens(self.scope, query)


def concept_lens(user_id: uuid.UUID, request: LensQuery) -> dict[str, object]:
    """Concept scores of the user's songs.

    ``{run_id, corpus_size, calibration, minimum_percentile, representation, scores}``, where
    ``scores`` maps each song id to the concepts it reaches.
    """
    requested = list(dict.fromkeys(name.strip() for name in request.concepts if name.strip()))
    definitions = {str(item["name"]).casefold(): item for item in predefined_concepts()}
    with connect() as connection:
        for item in curation.enabled_concepts(connection, user_id):
            definitions[str(item["name"]).casefold()] = item
    missing = [name for name in requested if name.casefold() not in definitions]
    if missing:
        raise NotFoundError(f"Unknown concepts: {', '.join(missing)}")
    rows, matrix, run_id = (
        lyrics_corpus(user_id) if request.representation == "lyrics" else semantic_corpus(user_id)
    )
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    lyrics_matrix: np.ndarray | None = None
    lyrics_available: np.ndarray | None = None
    if request.representation == "hybrid":
        lyrics_rows, available_lyrics, lyrics_run_id = lyrics_corpus(user_id)
        lyrics_by_id = {
            str(row["id"]): available_lyrics[index] for index, row in enumerate(lyrics_rows)
        }
        lyrics_matrix = np.zeros((len(rows), available_lyrics.shape[1]), dtype=np.float32)
        lyrics_available = np.asarray([str(row["id"]) in lyrics_by_id for row in rows], dtype=bool)
        for index, row in enumerate(rows):
            if lyrics_available[index]:
                lyrics_matrix[index] = lyrics_by_id[str(row["id"])]
        run_id = f"{run_id}+{lyrics_run_id}"
    row_index = {str(row["id"]): index for index, row in enumerate(rows)}
    result: dict[str, list[dict[str, object]]] = {str(row["id"]): [] for row in rows}
    for name in requested:
        definition = definitions[name.casefold()]
        positive_ids = definition.get("positive_track_ids") or []
        negative_ids = definition.get("negative_track_ids") or []
        positive_indices = [
            row_index[str(identifier)]
            for identifier in positive_ids
            if str(identifier) in row_index
        ]
        negative_indices = [
            row_index[str(identifier)]
            for identifier in negative_ids
            if str(identifier) in row_index
        ]
        positive_prompts = definition.get("positive_prompts") or []
        negative_prompts = definition.get("negative_prompts") or []
        semantic_evidence: np.ndarray | None = None
        lyrics_evidence: np.ndarray | None = None
        if request.representation == "lyrics":
            positive_parts = (
                list(shared_lyrics_model().embed_queries(positive_prompts))
                if positive_prompts
                else []
            )
            negative_parts = (
                list(shared_lyrics_model().embed_queries(negative_prompts))
                if negative_prompts
                else []
            )
            positive_parts.extend(matrix[positive_indices] if positive_indices else [])
            negative_parts.extend(matrix[negative_indices] if negative_indices else [])
            if not positive_parts:
                raise InvalidQueryError(f"Concept {name} has no usable lyrics evidence")
            positive = np.mean(positive_parts, axis=0)
            positive /= max(float(np.linalg.norm(positive)), 1e-8)
            raw = matrix @ positive
            if negative_parts:
                negative = np.mean(negative_parts, axis=0)
                negative /= max(float(np.linalg.norm(negative)), 1e-8)
                raw -= matrix @ negative
            percentiles = empirical_percentiles(raw)
            lyrics_evidence = percentiles
        else:
            semantic_raw, percentiles = score_concept(
                matrix,
                positive_prompts,
                negative_prompts,
                matrix[positive_indices] if positive_indices else None,
                matrix[negative_indices] if negative_indices else None,
            )
            raw = semantic_raw
            semantic_evidence = empirical_percentiles(semantic_raw)
            if (
                request.representation == "hybrid"
                and lyrics_matrix is not None
                and lyrics_available is not None
            ):
                positive_parts = (
                    list(shared_lyrics_model().embed_queries(positive_prompts))
                    if positive_prompts
                    else []
                )
                negative_parts = (
                    list(shared_lyrics_model().embed_queries(negative_prompts))
                    if negative_prompts
                    else []
                )
                positive_parts.extend(
                    lyrics_matrix[index] for index in positive_indices if lyrics_available[index]
                )
                negative_parts.extend(
                    lyrics_matrix[index] for index in negative_indices if lyrics_available[index]
                )
                if positive_parts:
                    positive = np.mean(positive_parts, axis=0)
                    positive /= max(float(np.linalg.norm(positive)), 1e-8)
                    lyrics_raw = lyrics_matrix @ positive
                    if negative_parts:
                        negative = np.mean(negative_parts, axis=0)
                        negative /= max(float(np.linalg.norm(negative)), 1e-8)
                        lyrics_raw -= lyrics_matrix @ negative
                    lyrics_evidence = empirical_percentiles(lyrics_raw, lyrics_available)
                    raw, percentiles = combine_concept_percentiles(
                        semantic_raw, lyrics_raw, lyrics_available
                    )
        for index, row in enumerate(rows):
            if percentiles[index] >= request.minimum_percentile:
                result[str(row["id"])].append(
                    {
                        "name": name,
                        "raw_score": float(raw[index]),
                        "percentile": float(percentiles[index]),
                        "semantic_percentile": float(semantic_evidence[index])
                        if semantic_evidence is not None
                        else None,
                        "lyrics_percentile": float(lyrics_evidence[index])
                        if lyrics_evidence is not None
                        else None,
                        "lyrics_available": bool(lyrics_available[index])
                        if lyrics_available is not None
                        else request.representation == "lyrics",
                    }
                )
    return {
        "run_id": run_id,
        "corpus_size": len(rows),
        "calibration": "empirical-library-percentile",
        "minimum_percentile": request.minimum_percentile,
        "representation": request.representation,
        "scores": result,
    }
