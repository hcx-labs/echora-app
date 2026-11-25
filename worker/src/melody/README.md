# Melody (`worker.melody`)

Melody contours for humming search ("hum a tune, find the song"). Each song gets a pitch
contour, sampled at 10 Hz, from each of three sources: the full mix, the Roformer vocals, and the
accompaniment (mix minus vocals). A hummed query is matched against them.

Contours are computed in two places:

- the library ingest's `melody` phase (`worker.library.ingest.MelodyPhase`), during
  `navidrome_sync` and `import`;
- the `melody` stage (`worker/src/stages.py`) of the `hum_corpus` operation, which indexes a
  user's songs into a hum corpus.

## Flow

`MelodyCorpusProcessor` (`corpus.py`) runs the `hum_corpus` batches, with progress phases
`preprocess` and `melody-index`:

- **plan**: checks that the corpus belongs to the user, then takes the batch's songs. The parent
  job has already created the corpus and capped it at `track_limit` songs (`DEFAULT_CORPUS_SIZE`,
  1–500).
- **open**: gets the contour run (`create_sync_run`) and points the corpus at it.
- **process**: a song that already has contours in the run is only added to the corpus. Otherwise
  it prepares the shared audio and extracts the contours (`track_contours`); on Modal, one song
  per call (`melody_one`).
- **persist**: stores the contours and adds the song to `hum_corpus_tracks`.

Its counters sit at the top level of the batch summary: `tracks` (songs added), `contours`
(contours stored), `failed`.

`index.py` holds the extraction, the stores, and the query side: `contour_index_key` and
`load_contours` (the user's index, which the backend's hum search service keeps in memory until
the key changes) and `match_hum` (contour extraction for the hum, coarse motif matching, then DTW
over the contours).

## What it stores

- `melody_contours`: track, run, source, hop, pitch and voiced flags.
- `analysis_runs` (kind `melody_contour`), keyed by `MELODY_CONTOUR_REVISION` and the extractor
  configuration. The revision includes `SEPARATION_REVISION`, so a new separator re-extracts
  every song.
- `hum_corpora` and `hum_corpus_tracks`: the user's corpus and its songs.

The ingest plan redoes a song whose contours are not from the current revision.

## Configuration

`MelodyConfig` (`config.py`), prefix `ECHORA_HUM_`:

- `max_workers`: the threads that score one query;
- `diagnostic_dir`: where opt-in query diagnostics go.

Numba compiles librosa's pitch helpers into `NUMBA_CACHE_DIR`, which `index.py` sets before
librosa is imported. See [runtime settings](../../../docs/runtime-settings.md).

## Failures

- A song whose audio cannot be fetched fails with `InputUnavailableError`.
- A separation or extraction error fails it with `ProcessingError("Melody extraction failed")`.
  This includes a song with no usable source (fewer than 30 voiced frames anywhere).
- Failed songs are not added to the corpus; building the corpus again retries them.
