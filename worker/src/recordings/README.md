# Recordings (`worker.recordings`)

Recording search: learned audio fingerprints that find the library song a short clip was
recorded from. This is opt-in and off until a verified model bundle is installed.

This package has two parts:

- **indexing**: the library ingest's `recording_fingerprint` phase
  (`worker.library.ingest.RecordingIndexPhase`), during `navidrome_sync` and `import`;
- **queries**: the `recording_search` job (`worker/src/operations.py` calls `search.execute`).

The `recordings` stage of `recordings_backfill` is a different thing: Chromaprint fingerprints
that group duplicate recordings (`worker.library.recordings`).

## Flow

Indexing, per song (`search.py`):

1. `needs_fingerprints`: is the song missing fingerprints for this encoder's
   `representation_id`? This also rejects bytes that differ from the song's identity hash.
2. `encode_fingerprints`: decode to mono 8 kHz and run the ONNX encoder (`encoder.py`), giving
   one unit-normalised 128-dimensional vector per 0.5 s segment.
3. `insert_fingerprints`: store them. The ingest counter is `recording_fingerprinted`.

A query (`search.execute`):

- reads the uploaded clip, which is private, expires, and runs under a time budget;
- encodes it;
- matches it against the fingerprints of the songs the user may see (`matcher.py`), track by
  track, by temporal consensus;
- stores the result.

Thresholds come from a calibrated match policy file, never from defaults. `calibration.py`
collects explicitly consented, labelled clips for calibration; it makes no recognition
decisions.

## What it stores

- `recording_fingerprints`: per track and `representation_id`. These are immutable: a new encoder
  bundle has a new representation, so the ingest plan fingerprints every song again for it.
- `recording_searches`: queries and their results. Audio is dropped after
  `audio_retention_seconds`, and the row after `result_retention_seconds`.
- `recording_diagnostic_captures` and `recording_calibration_samples`, each with an expiry.

The worker's maintenance loop removes expired rows and files.

## Configuration

`RecordingConfig` (`config.py`), prefix `ECHORA_RECORDING_`:

- the bundle: `model_manifest`, or `model_id` + `model_revision` + `model_directory`, provided by
  `python -m worker.models.download`;
- the match policy;
- ONNX threads;
- upload, queue and timeout limits;
- retention;
- calibration (`calibration_enabled` and its limits).

A blank manifest leaves search and indexing off. See
[runtime settings](../../../docs/runtime-settings.md).

## Failures

- `QueueFull` refuses a query when the search queue is full, or a calibration upload when
  calibration storage is full.
- `CalibrationDisabled` and `TrackUnavailable` refuse calibration uploads.
- A query that runs over its time budget fails with `ValueError`.
- During ingest, a song that cannot be indexed counts as `failed` and is retried by the next
  sync.

## See also

- [Recording search](../../../docs/recording-search.md)
