"""Whole-track seek-bar peaks, generated only by the sync pipeline."""

from __future__ import annotations

import numpy as np

from core.media.audio import decode_audio_channels

from core.revisions import WAVEFORM_REVISION

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


def store_waveform(connection, track_id, audio: bytes) -> None:
    from psycopg.types.json import Jsonb

    samples = decode_audio_channels(audio, sample_rate=SAMPLE_RATE)
    peaks = waveform_peaks(samples)
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO track_waveforms (track_id, revision, duration_seconds, peaks)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (track_id) DO UPDATE SET revision=EXCLUDED.revision,
                 duration_seconds=EXCLUDED.duration_seconds, peaks=EXCLUDED.peaks, created_at=now()""",
            (track_id, WAVEFORM_REVISION, len(samples) / SAMPLE_RATE, Jsonb(peaks)),
        )
