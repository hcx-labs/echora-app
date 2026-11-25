import numpy as np

from core.curation.journeys import select_journey, spherical_targets


def test_spherical_targets_keep_endpoints():
    start = np.array([1.0, 0.0], dtype=np.float32)
    end = np.array([0.0, 1.0], dtype=np.float32)
    targets = spherical_targets(start, end, 7)
    assert np.allclose(targets[0], start)
    assert np.allclose(targets[-1], end)
    assert np.allclose(np.linalg.norm(targets, axis=1), 1)


def test_journey_keeps_endpoints_and_avoids_recording_duplicates():
    angles = np.linspace(0, np.pi / 2, 8)
    tracks = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    targets = spherical_targets(tracks[0], tracks[-1], 6)
    groups = [None, "duplicate", "duplicate", None, None, None, None, None]
    steps = select_journey(tracks, targets, 0, 7, [f"artist-{i}" for i in range(8)], groups)
    indices = [step[0] for step in steps]
    assert indices[0] == 0
    assert indices[-1] == 7
    assert not ({1, 2} <= set(indices))


def test_journey_reserves_both_endpoint_recording_groups():
    angles = np.linspace(0, np.pi / 2, 8)
    tracks = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    targets = spherical_targets(tracks[0], tracks[-1], 6)
    groups = ["start", None, None, None, None, None, "end", "end"]

    steps = select_journey(
        tracks,
        targets,
        0,
        7,
        [f"artist-{i}" for i in range(8)],
        groups,
    )

    indices = [step[0] for step in steps]
    assert indices[0] == 0
    assert indices[-1] == 7
    assert 6 not in indices
