"""The library map: a 2-D projection of the user's songs, their communities and neighbours.

Communities are SNN-Leiden clusters, chosen over several resolutions by silhouette, seed
stability and plateau width. A map is stored as a community snapshot keyed by the corpus, so
the same corpus is served from the snapshot.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter

import igraph as ig
import numpy as np
from fastapi.encoders import jsonable_encoder
from sklearn.metrics import adjusted_rand_score, silhouette_score

from core.db.connection import connect
from core.embeddings import store as embeddings
from core.errors import InvalidQueryError
from core.library import store as library

COMMUNITY_SNAPSHOT_REVISION = 1


def cluster_embeddings(
    normalized: np.ndarray, similarities: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    count = len(normalized)
    if count < 3:
        return np.zeros(count, dtype=int), {
            "algorithm": "SNN-Leiden",
            "clusters": 1,
            "neighbors": max(0, count - 1),
        }
    neighbor_count = min(count - 1, max(5, round(np.sqrt(count))))
    nearest = np.argsort(similarities, axis=1)[:, -(neighbor_count + 1) : -1]
    neighbor_sets = [set(map(int, row)) for row in nearest]
    edge_weights: dict[tuple[int, int], float] = {}
    for left in range(count):
        for right_value in nearest[left]:
            right = int(right_value)
            edge = (min(left, right), max(left, right))
            shared = len(neighbor_sets[left] & neighbor_sets[right])
            weight = max(0.01, shared / neighbor_count) * max(
                0.01, float(similarities[left, right])
            )
            edge_weights[edge] = max(edge_weights.get(edge, 0.0), weight)
    graph = ig.Graph(n=count, edges=list(edge_weights), directed=False)
    weights = list(edge_weights.values())
    resolutions = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75]
    candidates: list[dict[str, object]] = []
    for resolution in resolutions:
        runs: list[tuple[np.ndarray, float]] = []
        for seed in range(6):
            ig.set_random_number_generator(random.Random(104729 + seed))
            partition = graph.community_leiden(
                objective_function="modularity",
                weights=weights,
                resolution=resolution,
                n_iterations=-1,
            )
            runs.append((np.asarray(partition.membership, dtype=int), float(partition.quality)))
        best_labels, quality = max(runs, key=lambda item: item[1])
        comparisons = [
            adjusted_rand_score(runs[left][0], runs[right][0])
            for left in range(len(runs))
            for right in range(left + 1, len(runs))
        ]
        stability = float(np.mean(comparisons)) if comparisons else 1.0
        clusters = len(np.unique(best_labels))
        silhouette = (
            float(silhouette_score(normalized, best_labels, metric="cosine"))
            if 1 < clusters < count
            else -1.0
        )
        candidates.append(
            {
                "resolution": resolution,
                "labels": best_labels,
                "clusters": clusters,
                "silhouette": silhouette,
                "stability": stability,
            }
        )
    has_multiple = any(int(candidate["clusters"]) > 1 for candidate in candidates)
    for index, candidate in enumerate(candidates):
        plateau = sum(
            index + offset in range(len(candidates))
            and candidates[index + offset]["clusters"] == candidate["clusters"]
            for offset in (-1, 1)
        )
        candidate["score"] = (
            float(candidate["silhouette"]) + 0.35 * float(candidate["stability"]) + 0.1 * plateau
        )
        if has_multiple and int(candidate["clusters"]) == 1:
            candidate["score"] = -10.0
    selected = max(candidates, key=lambda candidate: float(candidate["score"]))
    labels = np.asarray(selected["labels"], dtype=int)
    ordered = sorted(
        np.unique(labels), key=lambda label: (-int(np.sum(labels == label)), int(label))
    )
    remap = {old: new for new, old in enumerate(ordered)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
    return labels, {
        "algorithm": "SNN-Leiden",
        "clusters": len(ordered),
        "neighbors": neighbor_count,
        "resolution": selected["resolution"],
        "silhouette": selected["silhouette"],
        "seed_stability_ari": selected["stability"],
        "resolutions_tested": resolutions,
    }


def library_map(
    user_id, model: str = "muq_mulan", limit: int = 1500, semantic_weight: float | None = None
) -> dict[str, object]:
    """The map of the user's songs for one representation (``blend`` mixes MuQ and MERT)."""
    limit = min(max(limit, 10), 3000)
    if model not in {"muq_mulan", "mert", "lyrics", "blend"}:
        raise InvalidQueryError("Unknown representation")
    if model == "blend":
        semantic_weight = 0.5 if semantic_weight is None else semantic_weight
        if not 0 <= semantic_weight <= 1:
            raise InvalidQueryError("semantic_weight must be between 0 and 1")
    with connect() as connection:
        if model == "blend":
            rows = embeddings.blend_map_corpus(connection, user_id, limit)
        else:
            rows = embeddings.map_corpus(
                connection,
                user_id,
                "lyrics" if model == "lyrics" else "audio-track",
                "bge_m3" if model == "lyrics" else model,
                limit,
            )
    if not rows:
        return {"points": [], "model": model}
    if model == "blend":
        semantic = np.stack(
            [embeddings.parse_vector(row.pop("semantic_embedding")) for row in rows]
        )
        acoustic = np.stack(
            [embeddings.parse_vector(row.pop("acoustic_embedding")) for row in rows]
        )
        semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-8)
        acoustic /= np.maximum(np.linalg.norm(acoustic, axis=1, keepdims=True), 1e-8)
        matrix = np.concatenate(
            [np.sqrt(semantic_weight) * semantic, np.sqrt(1 - semantic_weight) * acoustic], axis=1
        )
    else:
        matrix = np.stack([embeddings.parse_vector(row.pop("embedding")) for row in rows])
    effective_weight = float(
        semantic_weight if model == "blend" else (1 if model == "muq_mulan" else 0)
    )
    corpus_material = "|".join(
        [
            model,
            f"{effective_weight:.4f}",
            str(COMMUNITY_SNAPSHOT_REVISION),
            *[f"{row['id']}:{row['run_id']}" for row in rows],
        ]
    )
    corpus_hash = hashlib.sha256(corpus_material.encode()).hexdigest()
    with connect() as connection:
        cached = library.cached_community_snapshot(
            connection, model, effective_weight, corpus_hash, COMMUNITY_SNAPSHOT_REVISION
        )
    if cached:
        payload = dict(cached["payload"])
        payload["snapshot_id"] = str(cached["id"])
        payload["snapshot_created_at"] = cached["created_at"]
        payload["cache_hit"] = True
        return payload
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    similarities = normalized @ normalized.T
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    left_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    projection = left_vectors[:, :2] * singular_values[:2]
    scale = np.percentile(np.abs(projection), 97, axis=0)
    projection = projection / np.maximum(scale, 1e-6)
    labels, clustering = cluster_embeddings(normalized, similarities)
    cluster_count = len(np.unique(labels))
    embedding_centers = np.stack(
        [normalized[labels == index].mean(axis=0) for index in range(cluster_count)]
    )
    embedding_centers /= np.maximum(np.linalg.norm(embedding_centers, axis=1, keepdims=True), 1e-8)
    membership_logits = normalized @ embedding_centers.T / 0.08
    membership_logits -= membership_logits.max(axis=1, keepdims=True)
    memberships = np.exp(membership_logits)
    memberships /= memberships.sum(axis=1, keepdims=True)
    if cluster_count:
        center_matrix = np.stack(
            [projection[labels == index].mean(axis=0) for index in range(cluster_count)]
        )
        layout_centers = center_matrix.copy()
        for _ in range(80):
            for left in range(cluster_count):
                for right in range(left + 1, cluster_count):
                    delta = layout_centers[right] - layout_centers[left]
                    distance = float(np.linalg.norm(delta))
                    if distance < 0.9:
                        if distance < 1e-6:
                            angle = (left * 2.399963 + right) % (2 * np.pi)
                            delta = np.array([np.cos(angle), np.sin(angle)])
                            distance = 1.0
                        shift = delta / distance * (0.9 - distance) * 0.08
                        layout_centers[left] -= shift
                        layout_centers[right] += shift
        packed = projection.copy()
        clustered = labels >= 0
        packed[clustered] = (
            layout_centers[labels[clustered]]
            + (projection[clustered] - center_matrix[labels[clustered]]) * 0.34
        )
        projection = packed
    projection -= projection.mean(axis=0, keepdims=True)
    packed_scale = np.percentile(np.abs(projection), 97, axis=0)
    projection /= np.maximum(packed_scale, 1e-6)
    center_matrix = (
        np.stack([projection[labels == index].mean(axis=0) for index in range(cluster_count)])
        if cluster_count
        else np.empty((0, 2))
    )
    communities = []
    for cluster in range(cluster_count):
        member_indices = np.flatnonzero(labels == cluster)
        genres = Counter(
            genre for item in member_indices for genre in (rows[item].get("genres") or []) if genre
        )
        top_genres = [{"name": name, "tracks": count} for name, count in genres.most_common(3)]
        representatives = sorted(
            member_indices,
            key=lambda item: float(normalized[item] @ embedding_centers[cluster]),
            reverse=True,
        )[:3]
        label_parts = [item["name"] for item in top_genres[:2]]
        label = " / ".join(label_parts) if label_parts else f"Community {cluster + 1}"
        cohesion = float(np.mean(normalized[member_indices] @ embedding_centers[cluster]))
        communities.append(
            {
                "id": cluster,
                "label": label,
                "x": float(center_matrix[cluster, 0]),
                "y": float(center_matrix[cluster, 1]),
                "size": len(member_indices),
                "cohesion": cohesion,
                "top_genres": top_genres,
                "representative_tracks": [
                    {
                        "id": str(rows[item]["id"]),
                        "title": rows[item]["title"],
                        "artist": rows[item].get("artist"),
                    }
                    for item in representatives
                ],
            }
        )
    nearest_count = min(6, max(0, len(rows) - 1))
    nearest_sets = [
        set(map(int, np.argsort(similarities[index])[::-1][1 : nearest_count + 1]))
        for index in range(len(rows))
    ]
    edge_map: dict[tuple[int, int], dict[str, object]] = {}
    for left, neighbors_for_left in enumerate(nearest_sets):
        for right in neighbors_for_left:
            edge = (min(left, right), max(left, right))
            edge_map[edge] = {
                "source_id": str(rows[edge[0]]["id"]),
                "target_id": str(rows[edge[1]]["id"]),
                "similarity": float(similarities[edge[0], edge[1]]),
                "mutual": left in nearest_sets[right],
                "cross_community": bool(labels[edge[0]] != labels[edge[1]]),
            }
    provenance = {
        "run_id": str(rows[0]["run_id"]),
        "model_revision": rows[0].get("model_revision"),
        "run_created_at": rows[0].get("run_created_at"),
        "track_count": len(rows),
    }
    points = []
    for index, row in enumerate(rows):
        cluster = int(labels[index])
        affinity = float(memberships[index, cluster])
        entropy = float(-np.sum(memberships[index] * np.log(np.maximum(memberships[index], 1e-12))))
        bridge_score = entropy / max(np.log(cluster_count), 1e-8) if cluster_count > 1 else 0.0
        strongest = np.argsort(memberships[index])[::-1][:3]
        cluster_memberships = [
            {"cluster": int(value), "strength": float(memberships[index, value])}
            for value in strongest
        ]
        nearest = np.argsort(similarities[index])[::-1]
        neighbors = [
            {
                "id": str(rows[item]["id"]),
                "title": rows[item]["title"],
                "artist": rows[item]["artist"],
                "similarity": float(similarities[index, item]),
            }
            for item in nearest
            if item != index
        ][:4]
        public_row = {
            key: value
            for key, value in row.items()
            if key not in {"run_id", "model_revision", "run_created_at"}
        }
        points.append(
            {
                **public_row,
                "x": float(projection[index, 0]),
                "y": float(projection[index, 1]),
                "cluster": cluster,
                "cluster_affinity": affinity,
                "cluster_memberships": cluster_memberships,
                "bridge_score": bridge_score,
                "neighbors": neighbors,
            }
        )
    payload = jsonable_encoder(
        {
            "points": points,
            "clusters": communities,
            "communities": communities,
            "edges": list(edge_map.values()),
            "model": model,
            "semantic_weight": effective_weight,
            "clustering": {**clustering, **provenance},
            "corpus_hash": corpus_hash,
        }
    )
    parameters = {
        "algorithm": clustering["algorithm"],
        "algorithm_revision": COMMUNITY_SNAPSHOT_REVISION,
        "neighbors": clustering["neighbors"],
        "resolution": clustering["resolution"],
        "resolutions_tested": clustering["resolutions_tested"],
        "semantic_weight": effective_weight,
    }
    metrics = {
        "silhouette": clustering["silhouette"],
        "seed_stability_ari": clustering["seed_stability_ari"],
    }
    with connect() as connection:
        snapshot = library.save_community_snapshot(
            connection,
            model,
            effective_weight,
            corpus_hash,
            COMMUNITY_SNAPSHOT_REVISION,
            len(rows),
            parameters,
            metrics,
            payload,
        )
    return {
        **payload,
        "snapshot_id": str(snapshot["id"]),
        "snapshot_created_at": snapshot["created_at"],
        "cache_hit": False,
    }
