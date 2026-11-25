import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from worker.library.visual_features import (
    BAND_COUNT,
    FRAME_LENGTH,
    HOP_LENGTH,
    HOP_SECONDS,
    MAX_DURATION_SECONDS,
    MAX_STRUCTURE_BINS,
    SAMPLE_RATE,
    VISUAL_FEATURE_REVISION,
    _structure,
    analyze_visual_features,
    extract_visual_features,
    store_visual_features,
)


def tone(frequency=440, seconds=2):
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.25 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_visual_features_are_aligned_finite_and_source_timed():
    samples = tone(seconds=1.2)
    result = extract_visual_features(samples)
    frames = result["frame_count"]
    assert result["revision"] == "2"
    assert frames == (len(samples) - 1) // HOP_LENGTH + 1
    assert result["hop_seconds"] == HOP_SECONDS
    assert result["duration_seconds"] == len(samples) / SAMPLE_RATE
    assert result["source_offset_seconds"] == 0
    assert (frames - 1) * HOP_SECONDS < result["duration_seconds"] <= frames * HOP_SECONDS
    for name in ("level", "centroid", "flux", "onset"):
        values = np.array(result[name])
        assert values.shape == (frames,)
        assert np.all((values >= 0) & (values <= 1))
    for name, width in [
        ("bands", BAND_COUNT),
        ("chroma", 12),
        ("pitch", 84),
        ("reactivity", 3),
        ("attacks", 3),
    ]:
        values = np.array(result[name])
        assert values.shape == (frames, width)
        assert np.all((values >= 0) & (values <= 1))
    assert np.array(result["waveform"]).shape == (frames, 64)
    assert all(len(values) == frames for values in result["timbre"].values())
    assert np.array(result["kernels"]["values"]).shape == (frames, 4)
    assert result["kernels"]["kind"] == "fixed_nonlearned"
    assert np.all(np.diff(result["band_centers_hz"]) > 0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("size", [1, 2, 511, 512, 2048, SAMPLE_RATE])
def test_short_silence_is_neutral_without_fabricated_pitch_or_beats(size):
    result = extract_visual_features(np.zeros(size, dtype=np.float32))
    for name in (
        "bands",
        "pitch",
        "chroma",
        "level",
        "onset",
        "flux",
        "waveform",
        "reactivity",
        "attacks",
    ):
        assert not np.any(result[name]), name
    assert not np.any(result["structure"]["chroma_cosine"])
    assert not np.any(result["structure"]["mfcc_rbf"])
    assert result["onset_times_seconds"] == result["beat_times_seconds"] == []
    assert result["tempo"]["bpm"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("size", [1, 3, 511, 512, 513])
def test_short_nonzero_inputs_have_safe_cqt_and_no_tempo(size):
    result = extract_visual_features(np.full(size, 0.1, dtype=np.float32))
    assert len(result["pitch"]) == (size - 1) // HOP_LENGTH + 1
    assert result["tempo"]["bpm"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("midi", [36, 57, 69, 81])
def test_pitch_resolved_cqt_and_chroma_match_tone(midi):
    result = extract_visual_features(tone(440 * 2 ** ((midi - 69) / 12)))
    # Avoid boundary padding while measuring dominant pitch.
    assert np.argmax(np.mean(result["pitch"][12:-12], axis=0)) + result["pitch_midi_start"] == midi
    assert np.argmax(np.mean(result["chroma"][12:-12], axis=0)) == midi % 12


def test_sustained_tone_does_not_invent_tempo():
    result = extract_visual_features(tone(seconds=4))
    assert result["tempo"]["bpm"] is None
    assert result["beat_times_seconds"] == []


def test_snapshots_are_actual_signed_source_samples_with_boundary_padding():
    samples = tone(330, 0.25)
    result = extract_visual_features(samples)
    offsets = np.array(result["waveform_offsets_samples"]) - FRAME_LENGTH // 2
    for index in (0, 4, result["frame_count"] - 1):
        positions = index * HOP_LENGTH + offsets
        expected = np.array([samples[p] if 0 <= p < len(samples) else 0 for p in positions])
        np.testing.assert_allclose(result["waveform"][index], expected, atol=0.000051)
    assert np.min(result["waveform"]) < 0 < np.max(result["waveform"])


@pytest.mark.parametrize("frequency,band", [(80, 0), (800, 1), (5000, 2)])
def test_reactivity_uses_physical_hz_bands(frequency, band):
    result = extract_visual_features(tone(frequency))
    assert result["reactivity_edges_hz"] == [20, 250, 2000, 10000]
    assert np.argmax(np.mean(result["reactivity"][10:-10], axis=0)) == band
    assert abs(np.median(result["timbre"]["centroid_hz"][10:-10]) - frequency) < 30


def test_discrete_click_onsets_beats_and_half_double_ambiguity_are_source_timed():
    samples = np.zeros(SAMPLE_RATE * 8, dtype=np.float32)
    expected = np.arange(0.5, 7.6, 0.5)
    k = np.arange(220)
    click = np.sin(2 * np.pi * 1800 * k / SAMPLE_RATE) * np.exp(-k / 40)
    for time in expected:
        start = round(time * SAMPLE_RATE)
        samples[start : start + len(click)] = click
    result = extract_visual_features(samples)
    onsets = np.array(result["onset_times_seconds"])
    assert len(onsets) == len(expected)
    assert np.max(np.abs(onsets - expected)) < 2 * HOP_SECONDS
    beats = np.array(result["beat_times_seconds"])
    assert len(beats) >= 12
    assert np.max(np.min(np.abs(beats[:, None] - expected), axis=1)) < 2 * HOP_SECONDS
    for times in (onsets, beats):
        assert np.all(np.diff(times) > 0)
        assert np.all((times >= 0) & (times < 8))
    tempo = result["tempo"]
    assert abs(tempo["bpm"] - 120) < 4
    assert tempo["half_double_tempo_ambiguity"] is True
    assert tempo["confidence_calibrated"] is False
    np.testing.assert_allclose(
        tempo["candidates_bpm"], np.array([0.5, 1, 2]) * tempo["bpm"], atol=0.002
    )


def test_structure_pools_before_quadratic_work_and_recognizes_recurrence():
    # 100k frames would require 40 GB for one full float32 frame recurrence.
    frames = 100_000
    mfcc = np.tile(np.arange(13)[:, None], (1, frames)).astype(float)
    chroma = np.zeros((frames, 12))
    chroma[:, 9] = 1
    result = _structure(mfcc, chroma, np.ones(frames, dtype=bool), frames * HOP_SECONDS)
    assert np.array(result["chroma_cosine"]).shape == (MAX_STRUCTURE_BINS, MAX_STRUCTURE_BINS)
    assert len(result["edges_seconds"]) == MAX_STRUCTURE_BINS + 1
    assert np.all(np.diff(result["edges_seconds"]) > 0)
    np.testing.assert_allclose(result["chroma_cosine"], 1)
    np.testing.assert_allclose(result["mfcc_rbf"], 1)
    # Alternating A/B timbre and pitch: A recurs at position 2, unlike B.
    mfcc = np.tile([0, 10, 0, 10], (13, 1))
    chroma = np.eye(12)[[0, 7, 0, 7]]
    result = _structure(mfcc, chroma, np.ones(4, dtype=bool), 8)
    assert result["chroma_cosine"][0][2] > result["chroma_cosine"][0][1]
    assert result["mfcc_rbf"][0][2] > result["mfcc_rbf"][0][1]


@pytest.mark.parametrize(
    "samples",
    [np.array([]), np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2)), np.array([101.0])],
)
def test_visual_features_reject_invalid_audio(samples):
    with pytest.raises(ValueError):
        extract_visual_features(samples)


def test_rate_and_duration_guard_precede_expensive_analysis():
    with pytest.raises(ValueError, match="22050"):
        extract_visual_features(np.zeros(5), 44100)
    with patch("worker.library.visual_features.librosa.stft") as stft:
        with pytest.raises(ValueError, match="600 seconds"):
            extract_visual_features(
                np.zeros(MAX_DURATION_SECONDS * SAMPLE_RATE + 1, dtype=np.float32)
            )
        stft.assert_not_called()


def test_store_uses_revision_two_and_source_hop_without_committing():
    connection = Mock()
    cursor = connection.cursor.return_value.__enter__ = Mock(return_value=Mock())
    connection.cursor.return_value.__exit__ = Mock(return_value=False)
    with patch("worker.library.visual_features.decode_audio", return_value=np.zeros(512)) as decode:
        result = store_visual_features(connection, "track", analyze_visual_features(b"audio"))
    decode.assert_called_once_with(b"audio", SAMPLE_RATE)
    sql, params = cursor.return_value.execute.call_args.args
    assert "ON CONFLICT (track_id)" in sql
    assert params[:4] == ("track", VISUAL_FEATURE_REVISION, result["duration_seconds"], HOP_SECONDS)
    connection.commit.assert_not_called()


def test_store_persists_terminal_unsupported_result_for_long_audio():
    connection = Mock()
    cursor = connection.cursor.return_value.__enter__ = Mock(return_value=Mock())
    connection.cursor.return_value.__exit__ = Mock(return_value=False)
    samples = np.zeros(MAX_DURATION_SECONDS * SAMPLE_RATE + 1, dtype=np.float32)
    with (
        patch("worker.library.visual_features.decode_audio", return_value=samples),
        patch("worker.library.visual_features.extract_visual_features") as extract,
    ):
        result = store_visual_features(connection, "track", analyze_visual_features(b"audio"))
    extract.assert_not_called()
    sql, params = cursor.return_value.execute.call_args.args
    assert "'unsupported'" in sql
    assert params == ("track", VISUAL_FEATURE_REVISION, len(samples) / SAMPLE_RATE, HOP_SECONDS)
    assert result == {
        "revision": VISUAL_FEATURE_REVISION,
        "status": "unsupported",
        "duration_seconds": len(samples) / SAMPLE_RATE,
    }
    connection.commit.assert_not_called()
