"""``GET /library/artists/similar``."""

from __future__ import annotations

import numpy as np
from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.api.v1.library._artist_models import ArtistFacet
from backend.library.artist_corpus import artist_embedding_corpus, artist_payload
from backend.library.artists import soft_chamfer_similarity, weighted_center
from core.errors import NotFoundError


class FacetMatch(ResponseModel):
    """The closest pair of facets between the two artists."""

    target_facet: int
    candidate_facet: int
    similarity: float


class SimilarArtist(ResponseModel):
    """An artist, their facets and how well they cover the target's facets."""

    artist: str
    track_count: int
    component_count: int
    facets: list[ArtistFacet]
    similarity: float
    target_coverage: float
    candidate_coverage: float
    coarse_similarity: float
    """Centroid similarity, used to shortlist candidates."""
    strongest_facet_match: FacetMatch


class SimilarArtists(ResponseModel):
    """Artists whose facets best match the target's, most similar first."""

    artist: str
    model: str
    results: list[SimilarArtist]


class ArtistsSimilar(Endpoint):
    """Artists that sound like the given artist."""

    method = "GET"
    path = "/artists/similar"
    response = SimilarArtists

    def handle(
        self,
        artist: str,
        model: str = "muq_mulan",
        limit: int = 12,
        user: SessionUser = Depends(current_user),
    ) -> dict:
        limit = min(max(limit, 1), 30)
        rows, matrix, _ = artist_embedding_corpus(model)
        groups: dict[str, list[int]] = {}
        names: dict[str, str] = {}
        for index, row in enumerate(rows):
            key = str(row["artist"]).casefold()
            groups.setdefault(key, []).append(index)
            names.setdefault(key, str(row["artist"]))
        target_key = artist.casefold()
        if target_key not in groups:
            raise NotFoundError("Artist not found")
        _, target = artist_payload(names[target_key], rows, matrix, groups[target_key])
        target_center = weighted_center(target)
        centroid_candidates = []
        for key, indices in groups.items():
            if key == target_key:
                continue
            center = np.mean(matrix[indices], axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-8)
            centroid_candidates.append((key, float(target_center @ center)))
        candidates = sorted(centroid_candidates, key=lambda item: item[1], reverse=True)[
            : max(limit * 3, 30)
        ]
        results = []
        for key, coarse_similarity in candidates:
            payload, profile = artist_payload(names[key], rows, matrix, groups[key])
            similarity, forward, backward, component_matrix = soft_chamfer_similarity(
                target, profile
            )
            target_component, candidate_component = np.unravel_index(
                np.argmax(component_matrix), component_matrix.shape
            )
            results.append(
                {
                    **payload,
                    "similarity": similarity,
                    "target_coverage": forward,
                    "candidate_coverage": backward,
                    "coarse_similarity": coarse_similarity,
                    "strongest_facet_match": {
                        "target_facet": int(target_component),
                        "candidate_facet": int(candidate_component),
                        "similarity": float(
                            component_matrix[target_component, candidate_component]
                        ),
                    },
                }
            )
        results.sort(key=lambda item: float(item["similarity"]), reverse=True)
        return {"artist": names[target_key], "model": model, "results": results[:limit]}
