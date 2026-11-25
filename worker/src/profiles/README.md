# Profiles (`worker.profiles`)

Audio profiles: each song's temporal sections and recurring musical modes, derived from the
window embeddings ingest stored. Profiles are computed in the database alone, with no audio or
GPU. They run at the end of the `library` stage for the batch's canonical tracks (summary key
`audio_profiles`), and as the `audio_profiles` operation (stage `audio_profiles`, no library
connection) for canonical track ids the user can see.

## Flow

`stages.profiles()` runs one `AudioProfileProcessor` (`pipeline.py`) per model in
`SUPPORTED_PROFILE_MODELS` (`muq_mulan`, `mert`):

1. `plan`: gets or creates the profile run for the current parameters, then selects the tracks
   whose current audio embedding has no profile in that run.
2. `process`: reads the stored `audio-window` embeddings and derives a `TrackAudioProfile`
   (`derive_audio_profile`). It finds temporal segments by dynamic programming over decorrelated
   windows, and modes by spherical k-means. When a song lacks exact window times, they are
   inferred from the window and stride. Once every song has a result, the run is marked
   complete.
3. `persist`: stores the profile in one transaction.

The progress phase is `audio_profiles`, and the processor stays quiet when nothing is planned.
Its summary is `{total, profiled, failed, model, profile_run_id}` (`facts()` adds the last two).
The stage nests these under `models` and adds `profiled` and `failed` totals.

## What it stores

`analysis_runs` (`kind='audio_profile'`), `track_audio_profiles`, `audio_temporal_segments`,
`audio_modes`, `audio_mode_intervals`. A track is profiled again when its source embedding run
changes, or when `AUDIO_PROFILE_REVISION` or the profile parameters
(`core.embeddings.representations`) change, which gives a new profile run.

## Configuration

The parameters come from `core.embeddings.representations` (`DEFAULT_PARAMETERS`). There is no
config class of its own. See [docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

A song without stored windows, or whose derivation fails, is yielded as a failed
`ProcessingError` ("Audio profile failed") and logged. The stage goes on, and the executor's
retry runs the batch again.

## See also

- [docs/audio-profiles.md](../../../docs/audio-profiles.md)
