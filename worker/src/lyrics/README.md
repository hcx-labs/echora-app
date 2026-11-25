# Lyrics (`worker.lyrics`)

A song's lyrics and their BGE-M3 embedding. Lyrics come from Navidrome. For a song without lyrics,
they are AI-transcribed from its vocals (see [`worker.transcription`](../transcription/README.md))
and then optionally refined by External AI. It runs as the `lyrics` stage (`worker/src/stages.py`)
in `navidrome_sync`, `import`, `lyrics_backfill` and `karaoke_backfill`. Translations, themes and
karaoke run after it and read what it stored.

## Flow

`lyrics_stage()` (`pipeline.py`) runs three processors in order and merges their summaries:

1. **`LyricsProcessor`**
   - **plan**: `plan_lyrics` (`worker/src/planning/audio.py`) selects songs without lyrics, or
     without a lyrics embedding of the current BGE-M3 revision. The `refresh_lyrics` job option
     selects every song.
   - **process**: reads each song's lyrics from Navidrome. The provider's copy is archived
     whatever happens. Manual lyrics and statuses stay authoritative, and a retrieval miss never
     erases earlier text. Clearly instrumental songs are stored as `instrumental`. The other songs
     without lyrics are transcribed when transcription is on, in phases: `preprocess` (one
     Roformer for every song), `transcription` (MOSS loaded once), then timing repair with the
     karaoke aligner for the drafts that need it.
   - **persist**: stores the lyrics and counts the song under its status.
2. **`TranscriptRefinementProcessor`**: the External AI second pass over this batch's new
   transcripts. It runs only when refinement is on. Its counters are `refined` and
   `refinement_failed`, and a transcript changed in the meantime is left alone.
3. **`LyricsEmbeddingProcessor`**: embeds every song with text using BGE-M3, loaded once
   (`models`); on Modal, every text goes in one call.

Progress phases: `lyrics`, `preprocess`, `transcription` and `models`. Summary `lyrics`:
`total`, `available`, `missing`, `unavailable`, `embedded` and `failed`, plus other statuses
(`instrumental`, ...) and the refinement counters once counted.

## What it stores

- `lyrics`: text, language, provenance and `availability_status`.
- `navidrome_lyrics_sources`: the provider's lyrics as read (`core.lyrics.sources.store_original`).
- `embeddings` (`embedding_type='lyrics'`): one whole-text vector, plus window vectors, under an
  `analysis_runs` row for the model revision. Each run records an attempt, with per-song results
  in `library.attempts`.

A song is redone when it has no lyrics text, or no whole-text lyrics embedding from a `bge_m3`
run of the configured revision (`ECHORA_EMBEDDING_LYRICS_REVISION`).

## Configuration

- `EmbeddingConfig` (`core.embeddings.config`, prefix `ECHORA_EMBEDDING_`) names the BGE-M3
  model and revision.
- Transcription needs `TranscriptionConfig` (see its README) and the `transcription` feature
  switch.
- Refinement uses the External AI settings stored in the database.

See [runtime settings](../../../docs/runtime-settings.md).

## Failures

- A retrieval, transcription or timing-repair error fails that song with a `ProcessingError`;
  the batch continues.
- An embedding failure is recorded in the run's attempt as "Lyrics embedding failed; see service
  logs", never with the lyrics or the error text.
- A refinement failure keeps the transcript and counts `refinement_failed`.
- Failed songs remain planned and are retried by the next run.

## See also

- [Song transcription](../../../docs/song-transcription.md)
- [Batch phases](../../../docs/batch-phases.md)
