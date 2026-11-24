"""Benchmark for the hum matching hot path.

Synthesizes contours sized like production data (4-minute tracks at 10 Hz,
10-20 s hum queries) and times query extraction prep, per-catalog matching,
and full search_corpus-equivalent scans. Timing only: every code path here is
the production one, and the correctness oracles in test_hum_search.py stay the
source of truth for matching behavior.
"""

from __future__ import annotations

import time

import numpy as np

from worker.melody.index import (
    _match_prepared_motifs,
    _motif_windows,
    _prepare_motifs,
    _relative,
    extract_hum_contour,
    match_contour,
)

CONTOUR_SECONDS = 240  # 4-minute track -> 2400 bins at 10 Hz
CATALOG_TRACKS = 50
SOURCES_PER_TRACK = 3


def _catalog_contour(rng: np.random.Generator, length: int) -> tuple[np.ndarray, np.ndarray]:
    # Smooth random walk in MIDI space with 15% unvoiced, like a real contour.
    steps = rng.normal(0, 0.35, length).cumsum().astype(np.float32)
    pitch = 62 + 6 * np.sin(np.linspace(0, rng.uniform(4, 30), length)) + steps * 0.2
    mask = rng.random(length) > 0.15
    return pitch.astype(np.float32), mask


def bench(label: str, fn, repeat: int = 5) -> float:
    fn()  # warmup (numba compile, caches)
    best = min(
        (lambda start=time.perf_counter(): (fn(), time.perf_counter() - start)[1])()
        for _ in range(repeat)
    )
    print(f"{label:55s} {best * 1000:9.1f} ms")
    return best


def main() -> None:
    rng = np.random.default_rng(42)

    print(
        f"catalog: {CATALOG_TRACKS} tracks x {SOURCES_PER_TRACK} sources, "
        f"{CONTOUR_SECONDS}s contours @10 Hz"
    )

    # --- Query-side prep --------------------------------------------------
    import io
    import wave

    samples = (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(15 * 24000)) / 24000) * 32767).astype(
        np.int16
    )
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(samples.tobytes())
        audio = buffer.getvalue()
    bench("extract_hum_contour (15 s synthetic tone)", lambda: extract_hum_contour(audio), repeat=3)

    query, query_mask = _catalog_contour(rng, 150)
    windows = _motif_windows(query, query_mask)
    print(f"query windows: {len(windows)}")
    bench("_prepare_motifs", lambda: _prepare_motifs(windows))
    prepared = _prepare_motifs(windows)

    # --- Single-contour match (one catalog source) ------------------------
    target, target_mask = _catalog_contour(rng, CONTOUR_SECONDS * 10)
    bench(
        "match_contour (single 2400-bin target)",
        lambda: match_contour(query, query_mask, target, target_mask),
    )
    bench(
        "_match_prepared_motifs (batched, 2400-bin target)",
        lambda: _match_prepared_motifs(prepared, target, target_mask),
    )

    # --- Full corpus scan (as in search_corpus thread pool) ---------------
    contours = [
        _catalog_contour(rng, CONTOUR_SECONDS * 10)
        for _ in range(CATALOG_TRACKS * SOURCES_PER_TRACK)
    ]

    def scan_serial() -> None:
        for target, target_mask in contours:
            _match_prepared_motifs(prepared, target, target_mask)

    elapsed = bench("full corpus scan, serial (150 contours)", scan_serial, repeat=3)
    print(f"{'  -> corpus scan latency budget per query':55s} {elapsed:9.2f} s")


if __name__ == "__main__":
    main()
