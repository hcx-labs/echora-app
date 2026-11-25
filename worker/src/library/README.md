# Library (`worker.library`)

Navidrome ingest: resolves each song of a batch to a canonical track from its audio bytes, then
runs every audio analysis the plan says the song still needs. It is the `library` stage
(`worker/src/stages.py`), the first stage of `navidrome_sync` and `import`. After ingest the
stage links the songs to the user (`linked`) and derives their audio profiles
(`audio_profiles`, see [profiles](../profiles/README.md)). The package also holds the
`recordings_backfill` stage's `FingerprintProcessor` (`recordings.py`).

## Flow

`ingest_navidrome()` (`ingest.py`) opens one `IngestBatch`: the database connection, the
Navidrome client, the library, and the `bindings` that tie each song to the track and the exact
bytes its identity was resolved from. Each later phase reads the song again and rejects other
bytes. Each phase is a processor over the songs it selects; it computes in `process` and writes
in `persist`, one transaction per song.

1. `IdentityPhase` (`identity`): downloads and hashes each song, then upserts the track and its
   source. Without hash checks (`verify_audio_hashes=false`) a known source keeps its track and
   only its metadata is refreshed.
2. `plan_audio()` (see [planning](../planning/README.md)), reported in the `planning` phase with
   the counts per analysis.
3. Audio preparation (`preprocess`): only when a prepared-audio cache is active. It decodes the
   formats the outstanding phases need, and the melody vocals locally, before any GPU model
   loads, holding one Roformer for the phase.
4. `WaveformPhase` (`waveform`), then `VisualFeaturePhase` (`visual_features`).
5. `EmbeddingPhase` for MuQ-MuLan (`muq`), then MERT (`mert`). Each loads its model once (progress
   phase `models`), records an analysis attempt, and on Modal uploads each song once and embeds
   the phase in one call. A song that already has the run's embedding is reused.
6. `MelodyPhase` (`melody`): melody contours for the hum search. On Modal, one separation call
   and one extraction call.
7. `FingerprintPhase` (`fingerprint`): Chromaprint, and grouping with a matching recording.
8. `RecordingIndexPhase` (`recording_fingerprint`): recording-search fingerprints with the
   configured encoder.
9. `DescriptorPhase` (`audio_descriptors`): sound descriptors.

Ingest ends with a `finalizing` update. `IngestSummary` keys (top level of the batch summary):
`requested`, `discovered`, `downloaded`, `inserted`, `already_linked`, `reused_embeddings`,
`embedded_muq`, `embedded_mert`, `fingerprinted`, `waveforms_generated`,
`visual_features_generated`, `melody_indexed`, `melody_contours`, `recording_matches`,
`recording_fingerprinted`, `described_audio`, `failed`.

`FingerprintProcessor` (`recordings_backfill`, phase `fingerprint`) fingerprints the batch's
songs that have no fingerprint; its counters `fingerprinted` and `matched` sit at the top level of
the summary.

## What it stores

- `libraries`, `tracks`, `track_sources`, `library_tracks`: identity. Tracks are keyed by content.
- `analysis_runs`, `embeddings`, `analysis_attempts` / `analysis_attempt_tracks`: audio embeddings
  and per-song attempt records.
- `track_waveforms` (`WAVEFORM_REVISION`), `track_visual_features` (`VISUAL_FEATURE_REVISION`),
  `track_audio_descriptors` (`DESCRIPTOR_REVISION`).
- `melody_contours` (`MELODY_CONTOUR_REVISION`), `track_fingerprints`, `recording_groups`,
  `recording_group_members`, `recording_match_evidence`, `recording_fingerprints`.

A phase redoes a song when its stored result is missing or from another revision, model revision
or encoder representation (the rules in `planning/audio.py`).

## Configuration

`LibraryConfig` (`ECHORA_LIBRARY_`): `fingerprint_timeout_seconds` (the fpcalc time limit) and
`source_recheck_seconds` (revalidate a source's bytes after this long; see
[planning](../planning/README.md) for how 0 behaves). The embedding models
come from `EmbeddingConfig` (`ECHORA_EMBEDDING_`). See
[docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

A song whose source cannot be read, or whose bytes change after identity, fails only in the
phases that read it. Each failed song adds to `failed`; the executor retries the batch, and the
plan selects only what is still missing. Embedding failures are recorded in the run's attempt
without the provider's message. A partial descriptor result is stored and counted as failed. A
song that cannot be prepared is left to its consumers, which count it themselves.

## See also

- [docs/batch-phases.md](../../../docs/batch-phases.md)
- [docs/audio-preprocessing.md](../../../docs/audio-preprocessing.md)
- [docs/audio-descriptors.md](../../../docs/audio-descriptors.md)
- [docs/visual-features-v2.md](../../../docs/visual-features-v2.md)
- [docs/recording-search.md](../../../docs/recording-search.md)
