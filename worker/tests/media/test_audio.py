import numpy as np

from worker.media.audio import (
    deterministic_windows,
    full_coverage_window_ranges,
    full_coverage_windows,
)


def test_short_audio_is_wrapped_to_one_window() -> None:
    source = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    windows = deterministic_windows(source, sample_rate=1, seconds=5)
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0], [1.0, 2.0, 3.0, 1.0, 2.0])


def test_long_audio_uses_three_fixed_windows() -> None:
    source = np.arange(100, dtype=np.float32)
    windows = deterministic_windows(source, sample_rate=1, seconds=10)
    assert [window[0] for window in windows] == [10, 45, 80]


def test_full_coverage_windows_overlap_and_include_the_exact_ending() -> None:
    source = np.arange(23, dtype=np.float32)
    windows = full_coverage_windows(source, sample_rate=1, seconds=10, stride_seconds=5)

    assert [window[0] for window in windows] == [0, 5, 10, 13]
    np.testing.assert_array_equal(windows[-1], np.arange(13, 23, dtype=np.float32))


def test_full_coverage_short_audio_still_produces_one_window() -> None:
    source = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    windows = full_coverage_windows(source, sample_rate=1, seconds=5, stride_seconds=2)
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0], [1.0, 2.0, 3.0, 1.0, 2.0])


def test_full_coverage_ranges_include_irregular_final_start() -> None:
    source = np.arange(23, dtype=np.float32)
    ranged = full_coverage_window_ranges(source, sample_rate=1, seconds=10, stride_seconds=5)

    assert [(start, end) for _, start, end in ranged] == [
        (0.0, 10.0),
        (5.0, 15.0),
        (10.0, 20.0),
        (13.0, 23.0),
    ]
