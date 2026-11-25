import numpy as np

from backend.library.artists import fit_artist_profile, soft_chamfer_similarity


def test_small_artist_uses_one_facet_per_track():
    tracks = np.eye(3, dtype=np.float32)
    profile = fit_artist_profile(tracks)
    assert len(profile.weights) == 3
    assert np.isclose(profile.weights.sum(), 1)


def test_soft_chamfer_detects_shared_artist_facet():
    target = fit_artist_profile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    shared = fit_artist_profile(np.array([[0.98, 0.02], [-1.0, 0.0]], dtype=np.float32))
    unrelated = fit_artist_profile(np.array([[-1.0, 0.0], [-0.8, -0.2]], dtype=np.float32))
    shared_score, _, _, _ = soft_chamfer_similarity(target, shared)
    unrelated_score, _, _, _ = soft_chamfer_similarity(target, unrelated)
    assert shared_score > unrelated_score
