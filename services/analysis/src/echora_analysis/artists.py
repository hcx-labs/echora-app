from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ArtistProfile:
    weights: np.ndarray
    centers: np.ndarray
    component_labels: np.ndarray


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def fit_artist_profile(embeddings: np.ndarray, maximum_components: int = 4) -> ArtistProfile:
    matrix = normalize_rows(embeddings)
    count = len(matrix)
    if count == 0:
        raise ValueError("An artist profile needs at least one track")
    if count < 5:
        return ArtistProfile(
            weights=np.full(count, 1 / count, dtype=np.float32),
            centers=matrix,
            component_labels=np.arange(count, dtype=int),
        )
    maximum = max(1, min(maximum_components, count // 5))
    best: GaussianMixture | None = None
    best_bic = float("inf")
    for components in range(1, maximum + 1):
        candidate = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            max_iter=150,
            n_init=3,
            random_state=42,
            reg_covar=1e-5,
        ).fit(matrix)
        bic = float(candidate.bic(matrix))
        if bic < best_bic:
            best, best_bic = candidate, bic
    if best is None:
        raise RuntimeError("Could not fit artist profile")
    return ArtistProfile(
        weights=best.weights_.astype(np.float32),
        centers=normalize_rows(best.means_),
        component_labels=best.predict(matrix),
    )


def weighted_center(profile: ArtistProfile) -> np.ndarray:
    value = np.sum(profile.weights[:, None] * profile.centers, axis=0, dtype=np.float64)
    return normalize_rows(value[None, :])[0]


def soft_chamfer_similarity(
    left: ArtistProfile, right: ArtistProfile
) -> tuple[float, float, float, np.ndarray]:
    similarities = np.clip(normalize_rows(left.centers) @ normalize_rows(right.centers).T, -1, 1)
    forward = float(np.sum(left.weights * similarities.max(axis=1)))
    backward = float(np.sum(right.weights * similarities.max(axis=0)))
    return (forward + backward) / 2, forward, backward, similarities


def representative_indices(embeddings: np.ndarray, profile: ArtistProfile) -> list[int]:
    matrix = normalize_rows(embeddings)
    representatives: list[int] = []
    for component, center in enumerate(profile.centers):
        members = np.flatnonzero(profile.component_labels == component)
        if not len(members):
            members = np.arange(len(matrix))
        representatives.append(int(members[np.argmax(matrix[members] @ center)]))
    return representatives
