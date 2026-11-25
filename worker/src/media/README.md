# Media (`worker.media`)

Audio decoding, the content-addressed cache of prepared audio, and Mel-Band-Roformer vocal
separation. It is not a stage: library ingest, lyrics transcription, melody and the other audio
consumers call it.

## Flow

- `audio.py`: ffmpeg decoding to float32 mono (`decode_audio`) or multi-channel
  (`decode_audio_channels`), and the full-coverage embedding windows.
- `preprocessing.py`: `preprocessing_session()` opens the batch's `AudioArtifactCache`; the
  operations runner opens one per job. While a session is active:
  - decodes and derived arrays are cached by source bytes and recipe;
  - entries are published atomically under a file lock, so two consumers never compute the same
    entry;
  - every consumer gets its own copy of the array.
  `prepare_audio()` fills the cache ahead of time with the mono and stereo rates, vocals
  (`vocal_waveform`), melody stems (`melody_waveforms`: vocals and mix-minus-vocals) or the
  vocal reference a later phase will read. `get_check()` returns the session's cancellation
  check.
- `roformer.py`: local Mel-Band-Roformer separation of 44.1 kHz stereo audio.
  `separation_phase()` keeps one model loaded for a batch phase, loaded on first use only;
  outside a phase each call loads and releases its own. `using()` lends an already loaded
  `Separator` (a Modal container's). The model code is vendored under `vendor/roformer`, with its
  notice.

## What it stores

No database rows. Cache entries (`.npy` plus a JSON manifest with the source digest and recipe,
never credentials) live in the cache directory; `prune()` keeps them within the size and age
limits. A recipe names `CACHE_SCHEMA`, `DECODE_REVISION` and, for separated audio,
`SEPARATION_REVISION` (`core/src/revisions.py`), so a change there makes a new entry rather than
reusing an old one. `roformer` re-exports `SEPARATION_REVISION` for that purpose.

## Configuration

`PreprocessConfig` (`ECHORA_PREPROCESS_`): `dir`, `max_bytes` (0: no limit), `ttl_seconds`,
`lock_timeout_seconds`, `lock_poll_seconds`. The separator model is pinned in
`core/src/revisions.py` and downloaded by `worker.models.download`. See
[docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

- Decoding failures raise to the caller, which counts the song as failed.
- Waiting longer than `lock_timeout_seconds` for another process's entry raises `TimeoutError`.
- `separate_vocals` rejects audio that is not finite float32 stereo, and raises when the
  checkpoint is not in the local cache (it never downloads).
- Cancellation propagates through every step.

## See also

- [docs/audio-preprocessing.md](../../../docs/audio-preprocessing.md)
- [docs/batch-phases.md](../../../docs/batch-phases.md)
