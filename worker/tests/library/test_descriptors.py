import json

import numpy as np
import pytest

from worker.library.descriptors import SAMPLE_RATE, analyze_waveform, energy_curve, spectral_curve


def tone(seconds=4, amplitude=0.25, frequency=440):
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    mono = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return np.column_stack([mono, mono])


def test_silence_has_no_fabricated_tempo_key_or_infinite_json():
    result = analyze_waveform(np.zeros((SAMPLE_RATE, 2), dtype=np.float32))
    assert result["sample_peak_dbfs"] is None
    assert result["rhythm"] is None
    assert result["key"] is None
    assert result["warnings"] == ["silent_audio"]
    json.dumps(result, allow_nan=False)


def test_levels_are_channel_aware_and_last_window_keeps_actual_duration():
    stereo = tone(1.25)
    stereo[:, 1] *= -1
    curve = energy_curve(stereo, SAMPLE_RATE)
    assert len(curve) == 2
    assert curve[-1]["end_seconds"] == 1.25
    assert curve[0]["rms_dbfs"] == pytest.approx(-15.0515, abs=0.01)
    assert curve[0]["sample_peak_dbfs"] == pytest.approx(-12.0412, abs=0.01)


def test_spectral_centroid_tracks_tone_frequency():
    stereo = tone(1, frequency=1000)
    curve = spectral_curve(stereo[:, 0], SAMPLE_RATE)
    assert curve[0]["centroid_hz"] == pytest.approx(1000, abs=30)


def test_ebu_loudness_tracks_attenuation_without_claiming_true_peak():
    loud = analyze_waveform(tone())
    quiet = analyze_waveform(tone(amplitude=0.125))
    assert loud["failed_components"] == []
    assert quiet["failed_components"] == []
    assert loud["loudness"]["integrated_lufs"] - quiet["loudness"][
        "integrated_lufs"
    ] == pytest.approx(6.0206, abs=0.15)
    assert loud["true_peak_dbtp"] is None
    assert loud["confidence_calibrated"] is False
    json.dumps(loud, allow_nan=False)


def test_tempo_retains_raw_confidence_and_valid_timestamps():
    stereo = np.zeros((15 * SAMPLE_RATE, 2), dtype=np.float32)
    click = np.hanning(600).astype(np.float32)
    for start in range(0, len(stereo) - len(click), SAMPLE_RATE // 2):
        stereo[start : start + len(click), :] = click[:, None]
    result = analyze_waveform(stereo)
    assert "rhythm" not in result["failed_components"]
    assert result["rhythm"]["half_double_tempo_ambiguity"] is True
    assert all(0 <= time <= 15 for time in result["rhythm"]["beat_times_seconds"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("waveform", [np.zeros((0, 2)), np.zeros(10), np.full((10, 2), np.nan)])
def test_invalid_audio_is_rejected(waveform):
    with pytest.raises(ValueError):
        analyze_waveform(waveform)


def test_wrong_sample_rate_is_rejected():
    with pytest.raises(ValueError, match="44100"):
        analyze_waveform(np.ones((100, 2)), 16000)
