# Themes (`worker.themes`)

What a song is about, for curation's Themes search. External AI reads the lyrics once and writes
a short English description: the situation, who is addressed, and the narrator's stance. It also
gives 5–10 theme tags. That text is embedded with BGE-M3, and curation compares theme queries
against it instead of the raw lyrics (`core.curation.preview.theme_overlay`). It runs as the
`themes` stage (`worker/src/stages.py`) after the lyrics stage, in `navidrome_sync`, `import` and
`lyrics_backfill`.

## Flow

`ThemeProcessor` (`pipeline.py`), progress phase `themes`:

- **plan**: plans nothing unless External AI and its `lyric_themes` switch are on. Otherwise it
  selects (`_pending`) the batch's songs with lyrics that have:
  - no description yet;
  - a description of other lyrics (source checksum);
  - a description from another prompt revision;
  - or a failure older than `RETRY_FAILED_AFTER_DAYS` (7).
- **open**: prepares the BGE-M3 embedder (on Modal, one call per chunk).
- **process**: works in chunks of `CHUNK` (32) songs, describing `CONCURRENCY` (4) at a time,
  then embeds the chunk's descriptions (`search_text`). It checks the switches between chunks
  and stops when either goes off.
- **persist**: stores the description, tags and vector, or the failure code.

Summary `themes`: `described`, `failed`.

## What it stores

`track_lyric_themes`: one row per song with its source checksum, prompt revision, model, status,
error code, description, themes, embedding and embedding model. The prompt revision is
`LYRIC_THEMES_REVISION` (`core/src/revisions.py`), the first 16 hex digits of the system prompt's
SHA-256. The tests check it against the prompt, so a prompt change must update it, and that
re-describes every song.

## Configuration

- No environment settings of its own.
- The endpoint, model and key are the External AI settings in the database.
- `lyric_themes` switches the stage on.
- The embedding model is `EmbeddingConfig`'s BGE-M3 (prefix `ECHORA_EMBEDDING_`).

See [runtime settings](../../../docs/runtime-settings.md).

## Failures

- `ThemeError` carries a safe code (`endpoint_failure`, `invalid_response`,
  `incomplete_response`, ...). An embedding failure is `embedding_failed`.
- Either way the song is stored as failed with that code (outcome `failed` with an output), and
  it is retried after a week. Logs hold only the code, never lyrics, replies or keys.
