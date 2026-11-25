import numpy as np

import core.curation.concepts as concepts


def test_concept_scores_support_overlap_and_library_percentiles(monkeypatch):
    vectors = {
        "dreamy": np.array([[1.0, 0.0]], dtype=np.float32),
        "aggressive": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    monkeypatch.setattr(
        concepts, "embed_texts", lambda texts: np.vstack([vectors[text] for text in texts])
    )
    tracks = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]], dtype=np.float32)

    raw, percentiles = concepts.score_concept(tracks, ["dreamy"], ["aggressive"])

    assert raw[0] > raw[1] > raw[2]
    assert percentiles.tolist() == [1.0, 0.5, 0.0]


def test_hybrid_scores_keep_missing_lyrics_neutral():
    semantic = np.asarray([0.9, 0.5, 0.1], dtype=np.float32)
    lyrics = np.asarray([0.1, 0.9, 0.0], dtype=np.float32)
    available = np.asarray([True, True, False])

    combined, percentiles = concepts.combine_concept_percentiles(semantic, lyrics, available)

    assert np.isclose(combined[0], 0.45)
    assert np.isclose(combined[1], 0.55 + 0.45 * 0.5)
    assert np.isclose(combined[2], 0.55 * 0.5)
    assert percentiles[1] == 1.0


def test_positive_example_can_define_concept_without_text(monkeypatch):
    monkeypatch.setattr(concepts, "embed_texts", lambda texts: np.empty((0, 2), dtype=np.float32))
    tracks = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    raw, _ = concepts.score_concept(tracks, [], positive_examples=tracks[:1])

    assert raw[0] > raw[1]


def test_percentiles_average_ties_and_are_permutation_invariant():
    scores = np.asarray([0.1, 0.7, 0.7, 0.9], dtype=np.float32)
    expected = np.asarray([0, 0.5, 0.5, 1], dtype=np.float32)
    assert np.allclose(concepts.empirical_percentiles(scores), expected)
    permutation = np.asarray([2, 0, 3, 1])
    assert np.allclose(concepts.empirical_percentiles(scores[permutation]), expected[permutation])


def test_constant_singleton_and_unavailable_scores_are_neutral():
    assert concepts.empirical_percentiles(np.ones(4)).tolist() == [0.5] * 4
    assert concepts.empirical_percentiles(np.asarray([0.9])).tolist() == [0.5]
    assert concepts.empirical_percentiles(np.asarray([])).size == 0
    values = np.asarray([0.2, 0.9, np.nan, np.inf])
    mask = np.asarray([False, True, True, True])
    assert concepts.empirical_percentiles(values, mask).tolist() == [0.5] * 4
