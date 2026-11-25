# Translation (`worker.translation`)

Line-by-line translations of a song's lyrics, written by the configured External AI endpoint for
each language pair the administrator set up. It runs as the `translations` stage
(`worker/src/stages.py`) after the lyrics stage, in `navidrome_sync`, `import` and
`lyrics_backfill`. Source lyrics stay authoritative: a translation is only ever added next to
them.

## Flow

`TranslationProcessor` (`pipeline.py`), progress phase `translation`:

- **plan**: loads the External AI settings. With translation off, no model or no language pairs,
  it plans nothing. Otherwise one item per song and pair:
  - pairs whose source is the song's language;
  - for a song already in a target language, its lines in other languages (`foreign_lines`,
    using the aligner's line language identification), when there are at least
    `MIXED_MIN_LINES`, as a `mixed` pair.
- **process**: checks the settings before every request, so switching translation off stops the
  stage.
  - A ready translation for the same lines is `reused`.
  - Otherwise `client.py` asks the model, with a bounded, non-agentic request (no tools,
    redirects or proxies). For mixed songs, `keep_foreign` keeps only the translated foreign
    lines.
  - If the lyrics changed while the request ran, the result is dropped as `stale`.
- **persist**: `core.lyrics.translations.save_translation`.

Summary `translations`: `translated`, `reused`, `failed`, `declined`, `stale`.

## What it stores

`lyric_translations`: per song and language pair, the lines, the source checksum, the prompt
revision and the model. A translation is current while its source checksum matches the song's
lines. Otherwise it is marked `stale` when translations are next loaded, and the next run
translates the song again.

## Configuration

No environment settings of its own. The endpoint, model, key, prompt and language pairs are the
External AI settings, stored encrypted in the database and edited in Settings. See
[runtime settings](../../../docs/runtime-settings.md).

## Failures

- A model that declines (`declined`, `refusal`) is counted as `declined`, not as a failure.
- Any other `TranslationError` fails the song with `TranslationFailedError`, whose stage is the
  safe error code. No response bodies, keys or lyrics are logged.
- A failed or declined song has no stored translation, so a later sync tries it again.

## See also

- [`worker.lyrics`](../lyrics/README.md)
