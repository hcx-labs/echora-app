# Cached visualization contract — revision 2

## Scope and migration

`worker/src/library/visual_features.py` produces measured, non-learned DSP features from the shared decoded mono source. This replaces revision 1's invalid multi-octave CQT hop, 10 Hz timeline, power-spectrum onset input, arbitrary mel-index bass groups, and browser-generated sine waveform.

Fresh schemas include migration 037's `track_visual_features` JSONB cache. Existing Alembic installations must apply `0042_track_visual_features` before the planner or API queries that table. No new dependency is required. The cache stores a revision, processing status, source duration and hop. `VISUAL_FEATURE_REVISION = "2"` is imported by the API and `processing_plan.plan_audio`; revision-1 rows are pending, never served as revision 2. Normal audio processing regenerates only the visual cache when other analysis contracts are current. Sources longer than 600 seconds are recorded as terminally unsupported so later processing runs do not repeatedly download and decode them. The existing 22050 Hz mono prerequisite is reused. A successful upsert atomically replaces the old cache; failed extraction rolls back. Processing does not delete model results.

**Backfill is not performed by these edits.** Run the existing library audio-processing workflow after deploying compatible backend/frontend versions. Inspect its planned visual-feature count first; the general workflow can also schedule other missing pipelines, so it is not a visual-only API. For a deliberately visual-only maintenance job, select stale/missing revisions and call `store_visual_features(connection, track_id, audio_bytes)` inside the existing authenticated/authorized worker context, committing per successful track. No download/model trigger belongs in the GET endpoint. Avoid relabeling old JSON as revision 2.

## Time, coverage and bounds

- Source rate: 22050 Hz mono; hop: **512 samples / 22050 ≈ 23.22 ms (43.07 Hz)**.
- `frame_count = ceil(source_samples / 512)`. Frame `i` is centered at source time `i * hop_seconds`, offset zero. No extra endpoint frame. STFT boundaries are zero padded.
- `duration_seconds` is the exact decoded sample count / rate, not rounded catalog duration. `source_offset_seconds`, `frame_alignment`, `frame_length`, sample rate and hop are explicit.
- Full sources of **at most 600 seconds** are supported: at most 25,840 frames. Longer sources raise before spectral computation; they are **not silently truncated** and remain pending/neutral. The existing decode stage precedes this guard, so the guard is not a bound on decoder memory. Supporting long mixes needs a separate chunked/LOD contract rather than an unbounded full-resolution payload. Normal processing currently retries these unsupported tracks; there is no persisted “unsupported length” status.
- Structure is pooled to at most **192 bins before any pairwise calculation**. Frame-level quadratic matrices are never constructed. Payload and DSP costs are linear in source frames apart from this fixed-size structure matrix. JSON values are rounded to four decimals (Hz mostly two, event times six).
- A local 60-second synthetic two-tone smoke benchmark produced 2,584 frames, 30 structure bins and 3,110,231 compact JSON bytes in 0.739 seconds after import (peak process RSS 451,944 KiB). This is not a real-library performance guarantee; JSON and Python list overhead remain significant.

## Feature fields

All per-frame arrays have exactly `frame_count` rows. Unless otherwise noted, normalized values are in `[0,1]` and are **track-relative**, not calibrated loudness or cross-track probabilities.

| Field | Representation |
|---|---|
| `bands`, `band_centers_hz` | 24 log-mel power bands, 30–10000 Hz, 80 dB display range with `band_reference_db` and `band_dynamic_range_db`; actual triangular-filter center frequencies, not linear FFT bins |
| `level` | STFT RMS, scaled by track 99.5th percentile |
| `reactivity`, `attacks` | Three STFT band energies with physical edges `[20,250,2000,10000]` Hz; shared robust scale preserves band balance; positive frame-to-frame differences |
| `flux` | Positive log-magnitude spectral differences, robustly scaled |
| `onset` | Continuous onset strength from **log-mel dB**, not linear power; a 0.3 dB mean-change floor suppresses leakage/numerical beats on sustained tones |
| `onset_times_seconds` | Discrete peak-picked source-time events; not every above-threshold frame |
| `beat_times_seconds`, `tempo` | Librosa beat tracking on that onset envelope; nullable BPM; half/main/double candidates and explicit uncalibrated ambiguity. No tempo for silence, insufficient duration/events, or fewer than two beats |
| `pitch` | 84 CQT bins, MIDI 24 (C1) through 107 (B7), 12 bins/octave, fixed tuning=0. **Not a melody transcription** |
| `chroma` | Octave-folded CQT pitch energy, row-max normalized, C..B |
| `waveform` | 64 genuine signed PCM sample values per centered 2048-sample source window, clipped only for display to `[-1,1]`; `waveform_offsets_samples` gives positions relative to window start. Boundary samples are zero padded. No synthesized oscillator |
| `timbre` | Spectral centroid, bandwidth and 85% rolloff in Hz; flatness and zero-crossing rate. Legacy `centroid` is centroid/Nyquist |
| `kernels` | Fixed, **non-learned** temporal rise/fall and spectral slope/curvature summaries on log-mel display bands. Spectral slope is signed; absolute second differences are divided by two to keep curvature in `[0,1]`. These are not CNN activations or embeddings |
| `structure` | Equal-frame pooled MFCC1..13 (excluding energy coefficient), standardized with a floor, RBF similarity; pooled chroma cosine similarity; adjacent-bin dissimilarity `novelty`; explicit source-time bin edges |

CQT receives enough right padding for very short inputs, then is cropped to actual source frames. Silence (STFT RMS ≤1e-7) has zero musical/reactivity/timbre features and no beats, rather than arbitrary chroma or tempo. Input must be finite, nonempty mono at the stated sample rate; implausible PCM amplitudes >100 are rejected. Waveform samples preserve actual quiet audio independently of the silence gate.

Limitations: sparse PCM snapshots are for visualization, **not reconstructable audio**; subsampling can alias and may miss extrema. Mono mixing can cancel antiphase channels (the separate stereo waveform/descriptor pipeline retains those). STFT/onset timing is hop-quantized; CQT's low notes have longer effective windows. Tempo can still be half/double or wrong on complex music. Recurrence is a bounded timbre/harmony similarity summary, not learned sections or semantic song structure. Novelty values mark pooled-bin transitions, not precise segment boundaries.

## Authenticated read-time reuse

`GET /library/tracks/{track_id}/visual-features` checks session and `user_track_links` **before** reading cache/enrichment. Response:

```
{ track_id, status: "complete" | "pending", visual_features: row | null,
  enrichment: { descriptors: row | null, vocal_activity: row | null,
                melody: bounded_preview | null } }
```

The endpoint only SELECTs: current descriptor revision, vocal activity joined to current representation runs, and current-revision existing melody contours. It invokes no model, schedules no work and does not rewrite the visual cache. Melody previews are capped at 1024 points. Thus later descriptor/voice/melody completion is visible on the next track load without a visual rerun. There is no live enrichment polling while a track remains loaded.

The browser accepts descriptor rhythm only when complete, recognized, finite/sorted and duration-matched, otherwise retaining core cached rhythm. Vocal windows preserve unknown gaps/tails as null. Existing voiced melody previews retain null/unvoiced points and are labeled as a coarse preview, not per-frame CQT pitch. RetroTrain uses cached vocal activation (when known) for vocal-driven motion; it does not relabel mid-band energy as measured vocal activity.

## Player event

All player visualizers consume **`echora:visual-frame`**, containing:

- `trackId`, source-time `timestamp`, `frameIndex`, `active`, source revision/rate/hop/duration, band Hz and waveform sample-position metadata;
- real `bands`, signed `waveform`, physical-band levels/attacks and level;
- discrete `onset`/`beat` crossing flags, nullable `bpm`, `rhythmSource`, ambiguous tempo candidates;
- `features`: pitch, chroma, spectral/timbre scalars, onset envelope and named non-learned kernels;
- bounded-bin `structuralNovelty`, optional cached `enrichment` vocal activity and melody preview pitch/source.

Recurrence matrices are validated but not retained in the animation timeline or copied into events. `echora:audio-reactivity` is temporarily emitted with the same frame for the shell backdrop outside the player ownership boundary. The fake expanded FFT-bin/sine-wave events have been removed; no current consumers of those events remain. Root, Signal, Cloud and RetroTrain no longer estimate tempo independently from onset intervals.

Strict validation rejects legacy revisions, oversized timelines/structures, wrong widths or source hops, nonfinite/out-of-range values, malformed physical axes and unordered/duplicate event times. Missing/invalid/loading data remains neutral. Requests use abort plus request-identity and track guards, including reloads of the same track. Pause, seek, load, waiting, end, error and missing data reset transient history. Playback follows `audio.currentTime`, not wall time; event crossings use binary search and are suppressed on discontinuities, preventing replayed beats after seeks. Requests cannot populate a later track. Cached rows update on animation frames without running a browser FFT.

## Verification

- 30 Python extractor/storage tests: silent/short signals, CQT MIDI/chroma accuracy, physical band separation, real PCM sample identity, click/onset/beat source timing, sustained-tone tempo suppression, strict input guards, finite JSON, structural bound at 100,000 synthetic feature frames and recurrence contrast.
- 5 mocked API/planner tests execute the real route body (isolated from heavyweight application imports): authentication, visibility, read-only optional enrichment, pending data and revision/decode planning.
- Database integration test extended for revision-1 invalidation and read-time enrichment without reruns; requires a disposable `TEST_DATABASE_URL` and was skipped locally.
- Player tests cover dimensions/ranges, legacy/malformed data, discrete event crossings, seek/reset neutrality, descriptor provenance, waveform identity, bounded structure, enrichment gaps and unified metadata. Python-generated one-sample, silent, tone and noise payloads were additionally accepted by the TypeScript validator.
- Local Python execution used existing `data/venvs/qwen-asr/bin/python` (3.12.14, librosa 1.0.0, NumPy 2.5.3) and existing pure-Python pytest 9.1.1 from the journal environment, appended to `sys.path`. No dependencies were installed. **The project-pinned librosa 0.11.0 / NumPy 2.5.2 environment was not available**, so this is not an exact pinned-environment qualification.
- Broader selected Python suite: 89 passed, 12 database skips, 3 failures: two pre-existing descriptor tests need missing Essentia; an unrelated lyrics-scoping test expects a different final SELECT. No fixes to unrelated pipelines were made.
- Focused extractor/API/waveform/melody/preprocessing/planner regression run: **74 passed**, plus 3 subtests. Player Node run: **18 passed**.
- `npm run typecheck` and `npm run build` pass. `npm run lint` passes with one existing PlayerProvider exhaustive-deps warning. Player Node tests pass. No live playback/browser or database E2E verification was performed.
