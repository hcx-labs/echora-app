"""Whole-track seek-bar peaks, generated only by the sync pipeline."""

from __future__ import annotations

import numpy as np

from core.revisions import WAVEFORM_REVISION
from worker.media.audio import decode_audio_channels

SAMPLE_RATE = 24_000
PEAK_COUNT = 1024


def waveform_peaks(samples: np.ndarray, count: int = PEAK_COUNT) -> list[float]:
    """Equal-time energy envelope with a small transient contribution.

    A peak-only envelope becomes a solid block on limited/mastered recordings.
    RMS measures sustained energy across both channels without phase cancellation.
    Keep 10% sample peak for short attacks, then apply a fixed display contrast
    curve. A truly constant-energy recording remains flat, as it should.
    """
    if count < 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Waveform requires finite, nonempty samples and a positive bucket count")
    samples = np.asarray(samples, dtype=np.float64)
    buckets = np.array_split(samples, min(count, len(samples)))
    envelope = np.array(
        [0.9 * np.sqrt(np.mean(block * block)) + 0.1 * np.max(np.abs(block)) for block in buckets]
    )
    maximum = float(envelope.max())
    if maximum > 0:
        envelope = (envelope / maximum) ** 1.5
    return [round(float(value), 4) for value in envelope]


def analyze_waveform(audio: bytes) -> tuple[float, object]:
    """A song's duration and the peaks its waveform is drawn from."""
    samples = decode_audio_channels(audio, sample_rate=SAMPLE_RATE)
    return len(samples) / SAMPLE_RATE, waveform_peaks(samples)


def store_waveform(connection, track_id, waveform: tuple[float, object]) -> None:
    """Store what ``analyze_waveform`` measured."""
    from psycopg.types.json import Jsonb

    duration, peaks = waveform
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO track_waveforms (track_id, revision, duration_seconds, peaks)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (track_id) DO UPDATE SET revision=EXCLUDED.revision,
                 duration_seconds=EXCLUDED.duration_seconds, peaks=EXCLUDED.peaks, created_at=now()""",
            (track_id, WAVEFORM_REVISION, duration, Jsonb(peaks)),
        )
