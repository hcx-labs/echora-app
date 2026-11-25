# Voice (`worker.voice`)

Voice evidence with Essentia's MTG-Jamendo classifiers: whether a song is sung, and the lead
voice's gender, plus its vocal activity over time. It runs as the `voice` stage
(`worker/src/stages.py`) in `navidrome_sync`, `import` and `voice_backfill`. The lyrics stage
reads the activity to recognise instrumental songs.

## Flow

`VoiceProcessor` (`pipeline.py`), progress phases `models` and `voice`:

- **plan**: gets or creates the `analysis_runs` row of the current classifier settings, then
  selects the batch's songs without both a voice vector and vocal activity from that run.
- **open**: on Modal it uploads every song once and classifies them all in one call, with
  results streaming back in order. Locally it loads the shared ONNX models once (`models`).
- **process**: classifies each song (`classify_with_activity`). Discogs-EffNet frame embeddings
  feed the gender and voice/instrumental heads. When every planned song has a result, it marks
  the run complete.
- **persist**: stores the vector and the activity.
- **close**: releases the model.

Summary `voice`: `total`, `classified`, `failed`.

## What it stores

- `embeddings` with `embedding_type='voice-gender'`: one 3-dimensional vector per song, the mean
  activations [instrumental, female, male], normalised to sum to one.
- `track_vocal_activity`: the per-window vocal activity, under the same run.

The run is keyed by model name (`mtg-jamendo-voice-gender-v2`), revision `joint-v2` and the hash
of `voice_config()` (`core.embeddings.representations`). A different configuration is a
different run, so every song is classified again.

## Configuration

`ModelsConfig.essentia_dir` (prefix `ECHORA_MODELS_`, older `ESSENTIA_MODELS_DIR`) is where the
three ONNX files must be. `python -m worker.models.download` provides them; nothing is
downloaded at inference time. See [runtime settings](../../../docs/runtime-settings.md).

## Failures

- A song whose audio cannot be fetched fails with `InputUnavailableError`.
- A classification error fails it with `ProcessingError("Voice classification failed")`.
- A missing model file raises `FileNotFoundError` naming the download command.
- Failed songs have no vector from the run, so the next run plans them again.
