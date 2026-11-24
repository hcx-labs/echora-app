"""Exact, track-at-a-time temporal consensus baseline (0.5-second hop).

Thresholds are caller-supplied operating points, not calibrated probabilities.
Inputs must contain unit-normalized 128-dimensional rows; zero rows denote
silence and never contribute evidence. Malformed/nonfinite inputs raise ValueError.
Offsets use reference_index - query_index, so negative offsets are valid.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

MATCHER_REVISION = "numpy-temporal-consensus-v3"
HOP_SECONDS = 0.5
MAX_MATCHES = 10


@dataclass(frozen=True)
class MatchPolicy:
    """Evidence thresholds and quality bands require representative evaluation.

    The legacy min_score setting marks the high-quality band, not a cutoff for
    returning candidates. Window support, coverage and spread still reject
    coincidences and unusable audio.

    Support counts distinct query fingerprints, not repeated windows. Coverage
    and score always use the *whole* query, including silence and missing overlap.
    """

    window_similarity: float
    min_score: float
    min_support: int
    min_coverage: float
    min_temporal_spread: float
    min_margin: float
    repeat_similarity: float = 0.995
    shortlist_size: int = 8
    offset_radius: int = 1
    block_size: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "window_similarity", "min_score", "min_coverage",
            "min_temporal_spread", "min_margin", "repeat_similarity",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name, minimum in (
            ("min_support", 2), ("shortlist_size", 2),
            ("offset_radius", 0), ("block_size", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.offset_radius > 4 or self.shortlist_size > 64 or self.block_size > 65536:
            raise ValueError("Matcher search bounds exceed supported limits")


def _vectors(value: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 128:
        raise ValueError(f"{name} must be an ndarray of shape (windows, 128)")
    if value.dtype.kind not in "fi" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real vectors")
    vectors = value.astype(np.float64, copy=False)
    norms = np.linalg.norm(vectors, axis=1)
    active = norms > 0
    if not np.all(np.isclose(norms[active], 1, atol=1e-5, rtol=1e-5)):
        raise ValueError(f"{name} nonzero vectors must be unit-normalized")
    return vectors, active


def match_recording(
    query: np.ndarray,
    references: Iterable[dict],
    policy: MatchPolicy,
    check: Callable[[], None] = lambda: None,
) -> dict:
    """Return state and at most MAX_MATCHES qualifying tracks, sorted deterministically.

    Reference dictionaries have a unique nonempty string track_id, fingerprints,
    and optional recording_group_id. Known equal groups waive the inter-track
    margin only; identities are never merged. Unknown groups remain competitors.
    Same-track competing offsets also produce ambiguity. Cancellation exceptions
    from check propagate. Scratch similarity storage is one reference block;
    votes are O(query length + current track length), never full-corpus arrays.
    """
    check()
    q, active = _vectors(query, "query")
    n = len(q)
    distinct: list[int] = []
    occurrences: list[list[int]] = []
    for i in np.flatnonzero(active):
        check()
        repeated = False
        for start in range(0, len(distinct), policy.block_size):
            check()
            hits = np.flatnonzero(q[distinct[start:start + policy.block_size]] @ q[i]
                                  >= policy.repeat_similarity)
            if len(hits):
                occurrences[start + int(hits[0])].append(int(i))
                repeated = True
                break
        if not repeated:
            distinct.append(int(i))
            occurrences.append([int(i)])
    if len(distinct) < policy.min_support:
        return {"state": "insufficient_audio", "matches": []}

    matches = []
    leaders = []  # Best representatives of the two strongest distinct identities.
    seen = set()

    def rank(item):
        return (-item["score"], item["track_id"], item["offset_seconds"])

    def identity(item):
        group = item["recording_group_id"]
        return ("group", group) if group is not None else ("track", item["track_id"])
    for reference in references:
        check()
        track_id = reference["track_id"]
        group = reference.get("recording_group_id")
        if not isinstance(track_id, str) or not track_id or track_id in seen:
            raise ValueError("track_id must be a unique nonempty string")
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError("recording_group_id must be None or a nonempty string")
        seen.add(track_id)
        r, r_active = _vectors(reference["fingerprints"], "fingerprints")
        m = len(r)
        if not m:
            continue
        votes = np.zeros(n + m - 1, dtype=np.int64)
        # Score every signed alignment independently of the support threshold.
        # Otherwise a close rival entirely below window_similarity disappears
        # from the identity margin, allowing a false high-quality label.
        alignment_scores = np.zeros(n + m - 1, dtype=np.float64)
        retrieval_similarity = policy.window_similarity
        for i in np.flatnonzero(active):
            for start in range(0, m, policy.block_size):
                check()
                end = min(start + policy.block_size, m)
                similarities = np.clip(r[start:end] @ q[i], -1, 1)
                alignment_scores[start - i + n - 1:end - i + n - 1] += np.maximum(similarities, 0)
                hits = np.flatnonzero((similarities >= retrieval_similarity)
                                      & r_active[start:end]) + start
                votes[hits - i + n - 1] += 1
        # Keep vote peaks for candidates plus exact score peaks for margin
        # competitors. Ties prefer smaller signed offsets in both rankings.
        ranked = set(np.argsort(-votes, kind="stable")[:policy.shortlist_size])
        ranked.update(np.argsort(-alignment_scores, kind="stable")[:policy.shortlist_size])
        offsets = set()
        for index in ranked:
            if alignment_scores[index] == 0:
                continue
            center = int(index) - n + 1
            offsets.update(range(max(1 - n, center - policy.offset_radius),
                                 min(m - 1, center + policy.offset_radius) + 1))
        evidence = []
        for offset in sorted(offsets):
            check()
            lo, hi = max(0, -offset), min(n, m - offset)
            similarities = np.zeros(n, dtype=np.float64)
            for start in range(lo, hi, policy.block_size):
                check()
                end = min(start + policy.block_size, hi)
                similarities[start:end] = np.clip(np.einsum(
                    "ij,ij->i", q[start:end], r[start + offset:end + offset]
                ), -1, 1)
            supported = similarities >= policy.window_similarity
            # Count repeated content once, but retain all supported occurrences
            # for timing and overlap. The first occurrence may be outside overlap.
            support = sum(bool(np.any(supported[indices])) for indices in occurrences)
            hits = np.flatnonzero(supported)
            coverage = float(len(hits) / n)
            spread = (hits[-1] - hits[0]) / (n - 1) if support > 1 else 0.0
            score = float(np.maximum(similarities, 0).sum() / n)
            evidence.append({
                "track_id": track_id, "recording_group_id": group,
                "score": score, "offset_seconds": offset * HOP_SECONDS,
                "support": support, "coverage": coverage,
                "temporal_spread": float(spread), "overlap_windows": hi - lo,
                "query_windows": n, "matched_windows": int(supported.sum()),
            })
        evidence.sort(key=lambda item: (-item["score"], item["offset_seconds"]))
        if evidence:
            # A runner-up just below an acceptance threshold still competes with
            # the winner. Filtering it first would exaggerate recognition margin.
            candidates = sorted(leaders + [evidence[0]], key=rank)
            leaders = []
            for candidate in candidates:
                if all(identity(candidate) != identity(item) for item in leaders):
                    leaders.append(candidate)
            del leaders[2:]
            qualifying = [item for item in evidence
                          if item["support"] >= policy.min_support
                          and item["coverage"] >= policy.min_coverage
                          and item["temporal_spread"] >= policy.min_temporal_spread]
            if qualifying:
                best = qualifying[0]
                best["offset_ambiguous"] = any(
                    best["score"] - other["score"] < policy.min_margin
                    for other in evidence if other is not best
                )
                matches.append(best)
                matches.sort(key=rank)
                del matches[MAX_MATCHES:]
    if not matches:
        return {"state": "no_match", "matches": []}
    winner = matches[0]
    runner = next((item for item in leaders if identity(item) != identity(winner)), None)
    margin = winner["score"] - runner["score"] if runner is not None else None
    winner["runner_up_margin"] = margin
    ambiguous = (winner["score"] < policy.min_score or winner["offset_ambiguous"]
                 or (margin is not None and margin < policy.min_margin))
    for index, item in enumerate(matches):
        item["match_quality"] = (
            "high" if index == 0 and not ambiguous
            else "possible" if item["score"] >= min(policy.window_similarity, policy.min_score)
            else "low"
        )
    return {"state": "ambiguous" if ambiguous else "identified", "matches": matches}
