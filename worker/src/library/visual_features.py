"""Versioned, measured DSP timelines; no browser FFT or learned feature claims."""

from __future__ import annotations

import librosa
import numpy as np

from core.library import store
from core.revisions import VISUAL_FEATURE_REVISION
from worker.media.audio import decode_audio

SAMPLE_RATE = 22_050
HOP_LENGTH = 512  # Divisible by 2**6 for the seven-octave CQT.
HOP_SECONDS = HOP_LENGTH / SAMPLE_RATE
FRAME_LENGTH = 2_048
BAND_COUNT = 24
MAX_DURATION_SECONDS = 600
MAX_STRUCTURE_BINS = 192
REACTIVITY_EDGES_HZ = [20, 250, 2_000, 10_000]


def _rounded(values: np.ndarray, digits: int = 4) -> list:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def _unit(values: np.ndarray) -> np.ndarray:
    """Track-relative robust scaling, preserving absolute silence."""
    scale = float(np.percentile(values, 99.5))
    return np.clip(values / scale, 0, 1) if scale > 1e-8 else np.zeros_like(values)


def _structure(mfcc: np.ndarray, chroma: np.ndarray, active: np.ndarray, duration: float) -> dict:
    """Pool before pairwise work: at most 192x192, never frames x frames."""
    count = min(MAX_STRUCTURE_BINS, max(1, int(np.ceil(duration / 2))), len(active))
    partitions = np.array_split(np.arange(len(active)), count)
    pooled_mfcc = np.stack([mfcc[:, indices].mean(axis=1) for indices in partitions])
    pooled_chroma = np.stack([chroma[indices].mean(axis=0) for indices in partitions])
    valid = np.array([active[indices].any() for indices in partitions])
    # Exclude MFCC0 (level); standardize timbral dimensions across the track.
    std = pooled_mfcc.std(axis=0)
    timbre = (pooled_mfcc - pooled_mfcc.mean(axis=0)) / np.maximum(std, 1.0)
    distance = np.maximum(
        0, (timbre**2).sum(1)[:, None] + (timbre**2).sum(1)[None, :] - 2 * timbre @ timbre.T
    )
    rbf = np.exp(-distance / (2 * mfcc.shape[0]))
    norms = np.linalg.norm(pooled_chroma, axis=1, keepdims=True)
    unit = pooled_chroma / np.maximum(norms, 1e-8)
    cosine = np.clip(unit @ unit.T, 0, 1)
    mask = valid[:, None] & valid[None, :]
    return {
        "method": "pooled_mfcc_rbf_chroma_cosine_nonlearned",
        "max_bins": MAX_STRUCTURE_BINS,
        "edges_seconds": _rounded(
            np.array([indices[0] * HOP_SECONDS for indices in partitions] + [duration]), 6
        ),
        "chroma_cosine": _rounded(cosine * mask),
        "mfcc_rbf": _rounded(rbf * mask),
        "novelty": _rounded(np.r_[0, 1 - ((cosine + rbf) / 2).diagonal(1)] * valid),
    }


def extract_visual_features(
    samples: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> dict[str, object]:
    """Analyze full mono audio up to ten minutes; reject rather than silently truncate.

    Frame i is centered at source sample i*512, with zero padding at boundaries.
    CQT is computed with additional right padding for extremely short input and
    cropped back to source frames. Silence is gated before relative normalization.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Visual features require 22050 Hz audio")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Visual features require finite, nonempty mono audio")
    if len(samples) > MAX_DURATION_SECONDS * SAMPLE_RATE:
        raise ValueError("Visual features support at most 600 seconds; no partial timeline stored")
    # Float PCM may exceed unity; clipping is confined to the display snapshots.
    if float(np.max(np.abs(samples))) > 100:
        raise ValueError("Audio amplitude exceeds supported PCM range")
    duration = len(samples) / sample_rate
    frame_count = (len(samples) - 1) // HOP_LENGTH + 1
    stft_samples = np.pad(samples, (0, max(0, FRAME_LENGTH - len(samples))))
    magnitude = np.abs(
        librosa.stft(
            stft_samples,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            center=True,
            pad_mode="constant",
        )
    )[:, :frame_count]
    power = magnitude**2
    rms = librosa.feature.rms(S=magnitude, frame_length=FRAME_LENGTH)[0]
    active = rms > 1e-7
    mel = librosa.feature.melspectrogram(
        S=power, sr=sample_rate, n_mels=BAND_COUNT, fmin=30, fmax=10_000
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=80)
    # Track-global reference, 80 dB display range. Silent frames remain exactly zero.
    band_reference_db = max(float(log_mel.max()), -80)
    bands = np.clip((log_mel - band_reference_db + 80) / 80, 0, 1).T
    bands[~active] = 0
    flux = _unit(np.r_[0, np.maximum(0, np.diff(np.log1p(magnitude), axis=1)).mean(axis=0)])
    onset_raw = librosa.onset.onset_strength(
        S=log_mel, sr=sample_rate, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH, center=True
    )
    # Ignore sub-dB leakage fluctuations before relative normalization; otherwise
    # a sustained pure tone can acquire a spurious beat from numerical noise.
    onset_raw = np.maximum(onset_raw - 0.3, 0)
    onset_raw[~active] = 0
    flux[~active] = 0
    level = _unit(rms)
    level[~active] = 0
    onset = _unit(onset_raw)
    onset_frames = np.array([], dtype=int)
    beat_frames = np.array([], dtype=int)
    bpm = None
    if active.any() and frame_count >= 4 and onset_raw.max() > 1e-8:
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_raw, sr=sample_rate, hop_length=HOP_LENGTH, backtrack=False
        )
        if duration >= 3 and len(onset_frames) >= 3:
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_raw, sr=sample_rate, hop_length=HOP_LENGTH, trim=True
            )
            candidate = float(np.asarray(tempo).reshape(-1)[0])
            if np.isfinite(candidate) and candidate > 0 and len(beat_frames) >= 2:
                bpm = candidate
    # C1..B7. Fixed tuning gives a reproducible MIDI mapping (not an inferred melody).
    pitch = np.zeros((frame_count, 84), dtype=np.float32)
    if active.any():
        padded = np.pad(samples, (0, max(0, 16_384 - len(samples))))
        cqt = np.abs(
            librosa.cqt(
                padded,
                sr=sample_rate,
                hop_length=HOP_LENGTH,
                fmin=librosa.midi_to_hz(24),
                n_bins=84,
                tuning=0,
                pad_mode="constant",
            )
        )[:, :frame_count].T
        pitch = _unit(cqt)
        pitch[~active] = 0
    chroma = pitch.reshape(frame_count, 7, 12).sum(axis=1)
    chroma /= np.maximum(chroma.max(axis=1, keepdims=True), 1e-8)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=FRAME_LENGTH)
    energies = np.stack(
        [
            np.sqrt(power[(frequencies >= lo) & (frequencies < hi)].sum(axis=0))
            for lo, hi in zip(REACTIVITY_EDGES_HZ[:-1], REACTIVITY_EDGES_HZ[1:])
        ],
        axis=1,
    )
    reactivity = _unit(energies)  # shared scale preserves bass/mid/treble balance
    reactivity[~active] = 0
    attacks = np.maximum(0, np.diff(reactivity, axis=0, prepend=reactivity[:1]))
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    zcr = librosa.feature.zero_crossing_rate(
        samples, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True
    )[0, :frame_count]
    for values in (centroid, bandwidth, rolloff, flatness, zcr):
        values[~active] = 0
    # Genuine signed PCM samples, not an oscillator or an inferred envelope.
    # 64 evenly spaced source samples in each 2048-sample centered window.
    padded_pcm = np.pad(samples, (FRAME_LENGTH // 2, FRAME_LENGTH // 2))
    offsets = np.round(np.linspace(0, FRAME_LENGTH - 1, 64)).astype(int)
    waveform = np.clip(padded_pcm[np.arange(frame_count)[:, None] * HOP_LENGTH + offsets], -1, 1)
    # Explicit fixed filters: temporal rise/fall, spectral slope and curvature.
    temporal = np.diff(bands, axis=0, prepend=bands[:1])
    slope = np.diff(bands, axis=1)
    curvature = np.diff(bands, n=2, axis=1)
    kernels = np.stack(
        [
            np.maximum(temporal, 0).mean(1),
            np.maximum(-temporal, 0).mean(1),
            slope.mean(1),
            np.abs(curvature).mean(1) / 2,
        ],
        axis=1,
    )
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=14)[1:]
    return {
        "revision": VISUAL_FEATURE_REVISION,
        "sample_rate": sample_rate,
        "hop_length": HOP_LENGTH,
        "hop_seconds": HOP_SECONDS,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "source_offset_seconds": 0,
        "frame_alignment": "centered_zero_padded",
        "frame_length": FRAME_LENGTH,
        "normalization": "track_relative_not_loudness_calibrated",
        "silence_rms_threshold": 1e-7,
        "band_reference_db": round(band_reference_db, 4),
        "band_dynamic_range_db": 80,
        "onset_floor_db": 0.3,
        "band_centers_hz": _rounded(
            librosa.mel_frequencies(n_mels=BAND_COUNT + 2, fmin=30, fmax=10_000)[1:-1], 2
        ),
        "bands": _rounded(bands),
        "level": _rounded(level),
        "centroid": _rounded(centroid / (sample_rate / 2)),
        "flux": _rounded(flux),
        "onset": _rounded(onset),
        "chroma": _rounded(chroma),
        "pitch": _rounded(pitch),
        "pitch_midi_start": 24,
        "pitch_bins_per_octave": 12,
        "waveform": _rounded(waveform),
        "waveform_offsets_samples": offsets.tolist(),
        "waveform_window_samples": FRAME_LENGTH,
        "reactivity_edges_hz": REACTIVITY_EDGES_HZ,
        "reactivity": _rounded(reactivity),
        "attacks": _rounded(attacks),
        "timbre": {
            "centroid_hz": _rounded(centroid, 2),
            "bandwidth_hz": _rounded(bandwidth, 2),
            "rolloff_hz": _rounded(rolloff, 2),
            "flatness": _rounded(flatness),
            "zcr": _rounded(zcr),
        },
        "onset_times_seconds": _rounded(onset_frames * HOP_SECONDS, 6),
        "beat_times_seconds": _rounded(beat_frames * HOP_SECONDS, 6),
        "tempo": {
            "bpm": round(bpm, 3) if bpm else None,
            "candidates_bpm": [round(bpm * factor, 3) for factor in (0.5, 1, 2)] if bpm else [],
            "half_double_tempo_ambiguity": True,
            "method": "librosa_onset_beat_track",
            "confidence_calibrated": False,
        },
        "kernels": {
            "kind": "fixed_nonlearned",
            "names": ["temporal_rise", "temporal_fall", "spectral_slope", "spectral_curvature"],
            "values": _rounded(kernels),
        },
        "structure": _structure(mfcc, chroma, active, duration),
    }


def analyze_visual_features(audio: bytes) -> dict[str, object]:
    """The visualizer's features of a song, or its duration when it is too long for them."""
    samples = decode_audio(audio, SAMPLE_RATE)
    duration = len(samples) / SAMPLE_RATE
    if duration > MAX_DURATION_SECONDS:
        return {
            "revision": VISUAL_FEATURE_REVISION,
            "status": "unsupported",
            "duration_seconds": duration,
        }
    return extract_visual_features(samples)


def store_visual_features(connection, track_id, features: dict[str, object]) -> dict[str, object]:
    """Store what ``analyze_visual_features`` measured."""
    if features.get("status") == "unsupported":
        store.save_unsupported_visual_features(
            connection, track_id, VISUAL_FEATURE_REVISION, features["duration_seconds"], HOP_SECONDS
        )
        return features
    store.save_visual_features(
        connection,
        track_id,
        VISUAL_FEATURE_REVISION,
        features["duration_seconds"],
        HOP_SECONDS,
        features,
    )
    return features
