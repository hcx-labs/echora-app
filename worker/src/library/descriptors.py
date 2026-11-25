"""Measured audio descriptors. These are not embedding scores or calibrated labels."""

from __future__ import annotations

import logging
import math

import numpy as np

from core.library import store
from core.revisions import DESCRIPTOR_REVISION
from worker.media.audio import decode_audio_channels

SAMPLE_RATE = 44_100
logger = logging.getLogger(__name__)


def finite_number(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def level_dbfs(amplitude: float) -> float | None:
    return 20.0 * math.log10(amplitude) if amplitude > 0 else None


def energy_curve(stereo: np.ndarray, sample_rate: int) -> list[dict[str, object]]:
    """One-second channel-energy windows, without cancellation from mono mixing."""
    curve = []
    for start in range(0, len(stereo), sample_rate):
        block = stereo[start : start + sample_rate].astype(np.float64)
        rms = float(np.sqrt(np.mean(block * block)))
        curve.append(
            {
                "start_seconds": start / sample_rate,
                "end_seconds": min(start + sample_rate, len(stereo)) / sample_rate,
                "rms_dbfs": level_dbfs(rms),
                "sample_peak_dbfs": level_dbfs(float(np.max(np.abs(block)))),
            }
        )
    return curve


def spectral_curve(mono: np.ndarray, sample_rate: int) -> list[dict[str, object]]:
    """One-second summaries of 2048-sample spectra with 50 percent overlap.

    Positive spectral flux describes timbral/rhythmic change, not beat confidence.
    """
    frame_size, hop = 2048, 1024
    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    buckets: dict[int, list[tuple[float, float]]] = {}
    previous = None
    for start in range(0, len(mono), hop):
        block = mono[start : start + frame_size]
        if len(block) < frame_size:
            block = np.pad(block, (0, frame_size - len(block)))
        spectrum = np.abs(np.fft.rfft(block * window))
        total = float(spectrum.sum())
        if total <= 1e-10:
            previous = None
            continue
        normalized = spectrum / total
        centroid = float(frequencies @ normalized)
        flux = float(np.maximum(normalized - previous, 0).sum()) if previous is not None else 0.0
        previous = normalized
        buckets.setdefault(start // sample_rate, []).append((centroid, flux))
    return [
        {
            "start_seconds": second,
            "end_seconds": min(second + 1, len(mono) / sample_rate),
            "centroid_hz": float(np.mean([item[0] for item in values])),
            "positive_spectral_flux": float(np.mean([item[1] for item in values])),
        }
        for second, values in sorted(buckets.items())
    ]


def analyze_waveform(stereo: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, object]:
    # RhythmExtractor2013's multifeature model expects 44.1 kHz.
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Audio descriptors require 44100 Hz audio")
    stereo = np.asarray(stereo, dtype=np.float32)
    if stereo.ndim != 2 or stereo.shape[1] != 2 or not len(stereo):
        raise ValueError("Audio descriptors require nonempty stereo audio")
    if not np.isfinite(stereo).all():
        raise ValueError("Audio contains non-finite samples")
    duration = len(stereo) / sample_rate
    mono = stereo.mean(axis=1)
    peak = float(np.max(np.abs(stereo)))
    result: dict[str, object] = {
        "revision": DESCRIPTOR_REVISION,
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "sample_peak_dbfs": level_dbfs(peak),
        "true_peak_dbtp": None,
        "rms_dbfs": level_dbfs(float(np.sqrt(np.mean(stereo.astype(np.float64) ** 2)))),
        "energy_curve": energy_curve(stereo, sample_rate),
        "spectral_curve": spectral_curve(mono, sample_rate),
        "loudness": None,
        "rhythm": None,
        "key": None,
        "warnings": [],
        "confidence_calibrated": False,
    }
    warnings = result["warnings"]
    if peak == 0:
        warnings.append("silent_audio")
        return result
    import essentia
    import essentia.standard as es

    result["extractor"] = {"name": "essentia", "version": essentia.__version__}
    # Partial outputs remain inspectable. Errors are explicit and retryable.
    errors = []
    try:
        if duration >= 3:
            _, _, integrated, loudness_range = es.LoudnessEBUR128(sampleRate=sample_rate)(stereo)
            result["loudness"] = {
                "integrated_lufs": finite_number(integrated),
                "range_lu": finite_number(loudness_range),
                "method": "EBU R128",
            }
        else:
            warnings.append("too_short_for_loudness_range")
    except Exception:
        logger.exception("Loudness extraction failed")
        errors.append("loudness")
    if duration >= 10 and np.any(mono):
        try:
            bpm, ticks, confidence, estimates, _ = es.RhythmExtractor2013(method="multifeature")(
                mono
            )
            valid_ticks = [
                float(t) for t in ticks if 0 <= t <= duration and math.isfinite(float(t))
            ]
            result["rhythm"] = {
                "bpm": finite_number(bpm) if bpm > 0 and confidence > 0 else None,
                "beat_times_seconds": valid_ticks,
                "confidence_raw": finite_number(confidence),
                "confidence_scale": "essentia_multifeature_uncalibrated",
                "bpm_estimates": [float(v) for v in estimates if math.isfinite(float(v))],
                "half_double_tempo_ambiguity": True,
            }
        except Exception:
            logger.exception("Rhythm extraction failed")
            errors.append("rhythm")
    else:
        warnings.append("insufficient_audio_for_tempo")
    if duration >= 3 and np.any(mono):
        try:
            key, scale, strength = es.KeyExtractor(sampleRate=sample_rate)(mono)
            result["key"] = {
                "tonic": key if strength > 0 else None,
                "scale": scale if strength > 0 else None,
                "strength_raw": finite_number(strength),
                "confidence_scale": "profile_correlation_uncalibrated",
                "scope": "whole_track",
                "key_changes_analyzed": False,
            }
        except Exception:
            logger.exception("Key extraction failed")
            errors.append("key")
    else:
        warnings.append("insufficient_audio_for_key")
    result["failed_components"] = errors
    return result


def analyze_audio_descriptors(audio: bytes) -> dict[str, object]:
    """A song's sound descriptors; ``failed_components`` lists those that could not be measured."""
    return analyze_waveform(decode_audio_channels(audio, SAMPLE_RATE, 2))


def store_audio_descriptors(connection, track_id, result: dict[str, object]) -> dict[str, object]:
    """Store what ``analyze_audio_descriptors`` measured (``partial`` with failed components)."""
    status = "partial" if result.get("failed_components") else "complete"
    store.save_audio_descriptors(connection, track_id, DESCRIPTOR_REVISION, status, result)
    return result
