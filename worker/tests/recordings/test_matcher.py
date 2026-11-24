"""Synthetic operating points test mechanics, not real-world calibration."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from worker.recordings.matcher import MATCHER_REVISION, MatchPolicy, match_recording


POLICY = MatchPolicy(
    window_similarity=0.85,
    min_score=0.65,
    min_support=6,
    min_coverage=0.65,
    min_temporal_spread=0.6,
    min_margin=0.08,
    block_size=7,
)


def vectors(n, seed=1):
    x = np.random.default_rng(seed).normal(size=(n, 128))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ref(x, track="a", group=None):
    return {"track_id": track, "fingerprints": x, "recording_group_id": group}


def test_exact_offset_and_generator():
    r = vectors(70)
    result = match_recording(r[13:33], (ref(r) for _ in range(1)), POLICY)
    assert result["state"] == "identified"
    best = result["matches"][0]
    assert best["offset_seconds"] == 6.5
    assert best["score"] == pytest.approx(1)
    assert best["support"] == 20
    assert best["coverage"] == 1


def test_noisy():
    r = vectors(60)
    q = r[17:37] + np.random.default_rng(4).normal(0, 0.015, (20, 128))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    result = match_recording(q, [ref(r)], POLICY)
    assert result["state"] == "identified"
    assert result["matches"][0]["offset_seconds"] == 8.5


def test_repeated_query_does_not_inflate_support():
    q = np.tile(vectors(2), (20, 1))
    assert match_recording(q, [ref(q)], POLICY)["state"] == "insufficient_audio"


def test_repeated_reference_sections_are_offset_ambiguous():
    q = vectors(20)
    result = match_recording(q, [ref(np.vstack([q, vectors(10, 9), q]))], POLICY)
    assert result["state"] == "ambiguous"
    assert result["matches"][0]["offset_seconds"] == 0
    assert result["matches"][0]["offset_ambiguous"]


def test_shared_window_cannot_identify_track():
    q = vectors(20)
    result = match_recording(q, [ref(np.tile(q[:1], (80, 1)))], POLICY)
    assert result == {"state": "no_match", "matches": []}


def test_matching_windows_without_temporal_consensus_do_not_match():
    q = vectors(20)
    assert match_recording(q, [ref(q[::-1])], POLICY)["state"] == "no_match"


def test_zero_reference_rows_preserve_offset():
    q = vectors(20)
    result = match_recording(q, [ref(np.vstack([np.zeros((4, 128)), q]))], POLICY)
    assert result["state"] == "identified"
    assert result["matches"][0]["offset_seconds"] == 2.0
    assert match_recording(q, [ref(np.zeros((30, 128)))], POLICY)["state"] == "no_match"


def test_unrelated():
    assert match_recording(vectors(20), [ref(vectors(100, 3))], POLICY) == {
        "state": "no_match",
        "matches": [],
    }


@pytest.mark.parametrize("q", [np.zeros((20, 128)), vectors(5), np.empty((0, 128))])
def test_insufficient(q):
    assert match_recording(q, [], POLICY)["state"] == "insufficient_audio"


@pytest.mark.parametrize(
    "q",
    [
        np.ones((20, 128)),
        np.ones((20, 127)),
        np.ones(128),
        np.full((20, 128), np.nan),
        np.full((20, 128), np.inf),
        np.ones((20, 128), dtype=complex),
    ],
)
def test_invalid_query(q):
    with pytest.raises(ValueError):
        match_recording(q, [], POLICY)


def test_invalid_reference():
    with pytest.raises(ValueError):
        match_recording(vectors(20), [ref(np.ones((20, 128)))], POLICY)


def test_group_duplicate_edition_and_distinct_runner():
    q = vectors(20)
    same = [ref(q, "b", "g"), ref(q, "a", "g")]
    result = match_recording(q, iter(same), POLICY)
    assert result["state"] == "identified"
    assert [m["track_id"] for m in result["matches"]] == ["a", "b"]
    assert result["matches"][0]["runner_up_margin"] is None
    for group in (None, "other"):
        result = match_recording(q, same + [ref(q, "c", group)], POLICY)
        assert result["state"] == "ambiguous"
        assert result["matches"][0]["runner_up_margin"] == pytest.approx(0)


def test_unknown_groups_compete_and_ties_are_order_independent():
    q = vectors(20)
    refs = [ref(q, "z"), ref(q, "a")]
    first = match_recording(q, refs, POLICY)
    assert first == match_recording(q, reversed(refs), POLICY)
    assert first["state"] == "ambiguous"
    assert first["matches"][0]["track_id"] == "a"


@pytest.mark.parametrize("offset", [-6, 0, 6])
def test_boundary_overlap_and_whole_query_denominator(offset):
    q = vectors(20)
    r = q[6:] if offset < 0 else np.vstack([vectors(offset, 9), q])
    result = match_recording(q, [ref(r)], POLICY)
    assert result["state"] == "identified"
    best = result["matches"][0]
    assert best["offset_seconds"] == offset * 0.5
    assert best["coverage"] == (0.7 if offset < 0 else 1)
    assert best["score"] == pytest.approx(best["coverage"])
    assert match_recording(q, [ref(q[:6])], POLICY)["state"] == "no_match"


def test_spread_is_required_independently():
    q = vectors(20)
    policy = replace(POLICY, min_score=0.3, min_coverage=0.3, min_temporal_spread=0.8)
    assert match_recording(q, [ref(q[7:14])], policy)["state"] == "no_match"


def test_silence_counts_against_coverage():
    q = np.vstack([vectors(10), np.zeros((10, 128))])
    assert match_recording(q, [ref(q)], POLICY)["state"] == "no_match"


def test_empty_reference_and_duplicate_ids():
    q = vectors(20)
    assert match_recording(q, [ref(np.empty((0, 128)))], POLICY)["state"] == "no_match"
    with pytest.raises(ValueError, match="unique"):
        match_recording(q, [ref(q), ref(q)], POLICY)


def test_cancellation_during_track_search():
    calls = 0

    class Cancelled(Exception):
        pass

    def check():
        nonlocal calls
        calls += 1
        if calls == 100:
            raise Cancelled

    with pytest.raises(Cancelled):
        match_recording(vectors(20), [ref(vectors(100))], POLICY, check)
    assert calls == 100


@pytest.mark.parametrize(
    "changes",
    [
        {"min_score": float("nan")},
        {"min_support": 0},
        {"block_size": 0},
        {"shortlist_size": 1},
        {"offset_radius": -1},
        {"min_margin": 0},
        {"min_score": "0.9"},
        {"min_score": True},
        {"min_support": 6.0},
        {"min_coverage": None},
        {"block_size": True},
    ],
)
def test_policy_validation(changes):
    with pytest.raises(ValueError):
        replace(POLICY, **changes)


def test_json_policy_and_revision():
    manifest = json.loads(json.dumps({"thresholds": asdict(POLICY)}))
    assert MatchPolicy(**manifest["thresholds"]) == POLICY
    assert MATCHER_REVISION == "numpy-temporal-consensus-v3"
    with pytest.raises(TypeError):
        MatchPolicy(**dict(manifest["thresholds"], unknown=1))


def test_bounded_results_do_not_hide_distinct_group_runner():
    q = vectors(20)
    refs = [ref(q, f"a{i:02}", "same") for i in range(15)]
    refs.append(ref(q, "z", "different"))
    result = match_recording(q, iter(refs), POLICY)
    assert len(result["matches"]) == 10
    assert result["state"] == "ambiguous"
    assert result["matches"][0]["runner_up_margin"] == pytest.approx(0)
    assert result == match_recording(q, reversed(refs), POLICY)


def test_runner_below_acceptance_threshold_still_competes():
    q = vectors(20)
    noise = vectors(20, seed=102)
    noise -= (noise * q).sum(axis=1, keepdims=True) * q
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    policy = replace(POLICY, window_similarity=0.7, min_score=0.9)
    winner = 0.93 * q + np.sqrt(1 - 0.93**2) * noise
    runner = 0.89 * q + np.sqrt(1 - 0.89**2) * noise
    result = match_recording(q, [ref(winner, "winner"), ref(runner, "runner")], policy)
    assert result["state"] == "ambiguous"
    assert result["matches"][0]["runner_up_margin"] == pytest.approx(0.04)


@pytest.mark.parametrize("score,windows,quality", [(0.57, 20, "possible"), (0.55, 14, "low")])
def test_score_below_high_quality_band_is_returned(score, windows, quality):
    q = vectors(20)
    noise = vectors(20, seed=102)
    noise -= (noise * q).sum(axis=1, keepdims=True) * q
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    r = score * q + np.sqrt(1 - score**2) * noise
    r[windows:] = 0
    policy = replace(POLICY, window_similarity=0.5, min_score=0.6)
    result = match_recording(q, [ref(r)], policy)
    assert result["state"] == "ambiguous"
    best = result["matches"][0]
    assert best["track_id"] == "a"
    assert best["score"] < policy.min_score
    assert best["match_quality"] == quality
    assert best["offset_seconds"] == 0


def test_quality_band_does_not_change_candidate_visibility():
    q = vectors(20)
    result = match_recording(q, [ref(q)], POLICY)
    assert result["matches"][0]["match_quality"] == "high"
    duplicate = match_recording(q, [ref(q, "a"), ref(q, "b")], POLICY)
    assert all(item["match_quality"] == "possible" for item in duplicate["matches"])


def test_runner_entirely_below_window_threshold_still_prevents_high_quality():
    q = vectors(20)
    noise = vectors(20, seed=102)
    noise -= (noise * q).sum(axis=1, keepdims=True) * q
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    winner = 0.90 * q + np.sqrt(1 - 0.90**2) * noise
    runner = 0.84 * q + np.sqrt(1 - 0.84**2) * noise
    result = match_recording(q, [ref(winner, "winner"), ref(runner, "runner")], POLICY)
    assert result["state"] == "ambiguous"
    assert result["matches"][0]["track_id"] == "winner"
    assert result["matches"][0]["runner_up_margin"] == pytest.approx(0.06)
    assert result["matches"][0]["match_quality"] == "possible"


def test_repeated_passage_keeps_supported_occurrences_for_spread():
    q = np.vstack([np.eye(128)[:10]] * 2)
    result = match_recording(q, [ref(q)], POLICY)
    assert result["state"] == "identified"
    best = result["matches"][0]
    assert best["support"] == 10
    assert best["coverage"] == best["temporal_spread"] == 1


def test_later_repetitions_can_support_partial_overlap():
    a, b = vectors(10), vectors(10, 42)
    q = np.vstack([a, b, a])
    policy = replace(POLICY, min_coverage=0.45, min_temporal_spread=0.45)
    result = match_recording(q, [ref(q[15:])], policy)
    assert result["matches"]
    best = result["matches"][0]
    assert best["offset_seconds"] == -7.5
    assert best["support"] == 15
    assert best["coverage"] == 0.5
