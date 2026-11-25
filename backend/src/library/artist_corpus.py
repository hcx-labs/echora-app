"""Every artist's songs and audio embeddings, for artist profiles and similar artists."""

from __future__ import annotations

import hashlib

import numpy as np

from backend.library.artists import fit_artist_profile, representative_indices
from core.db.connection import connect
from core.embeddings import store
from core.errors import ConflictError, InvalidQueryError


def artist_embedding_corpus(model: str) -> tuple[list[dict[str, object]], np.ndarray, str]:
    """Songs with an artist (id, title, artist, album), their embeddings and the corpus hash."""
    if model not in {"muq_mulan", "mert"}:
        raise InvalidQueryError("Artist profiles support muq_mulan or mert")
    with connect() as connection:
        rows = store.artist_corpus(connection, model)
    if not rows:
        raise ConflictError("No artist embeddings are available")
    matrix = np.stack([store.parse_vector(row.pop("embedding")) for row in rows])
    corpus_hash = hashlib.sha256(
        "|".join(f"{row['id']}:{row['run_id']}" for row in rows).encode()
    ).hexdigest()
    for row in rows:
        row.pop("run_id", None)
    return rows, matrix, corpus_hash


def artist_payload(
    name: str, rows: list[dict[str, object]], matrix: np.ndarray, indices: list[int]
) -> tuple[dict[str, object], object]:
    """An artist's facets (weight, song count, representative song) and the fitted profile."""
    artist_matrix = matrix[indices]
    profile = fit_artist_profile(artist_matrix)
    representatives = representative_indices(artist_matrix, profile)
    facets = []
    for component, representative in enumerate(representatives):
        members = np.flatnonzero(profile.component_labels == component)
        track = rows[indices[representative]]
        facets.append(
            {
                "index": component,
                "weight": float(profile.weights[component]),
                "track_count": int(len(members)),
                "representative_track": {
                    "id": str(track["id"]),
                    "title": track["title"],
                    "artist": track["artist"],
                    "album": track["album"],
                },
            }
        )
    return {
        "artist": name,
        "track_count": len(indices),
        "component_count": len(profile.weights),
        "facets": facets,
    }, profile
