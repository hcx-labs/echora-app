# Transcription (`worker.transcription`)

AI lyrics: a timed transcript of a song's separated vocals, made with a MOSS checkpoint. This
package has no stage of its own. The lyrics stage ([`worker.lyrics`](../lyrics/README.md)) uses
it for songs that Navidrome has no lyrics for, while the `transcription` feature is on at the
job's location.

## Flow

`SongTranscriber` (`song.py`) works in batch phases, which the lyrics stage drives across every
song that needs a transcript:

1. **Separation** (`preprocess`): one Roformer separates the vocals of every song
   (`worker.media`), so vocals are cached before MOSS loads.
2. **Generation** (`transcription`): MOSS loads once and transcribes every song in windows
   (`windows`, `parse_window`).
   - `guard.py` keeps generation to well-formed segments.
   - `recovery.py` and `budget.py` retry failed windows within song-wide limits and report the
     regions they could not resolve.
   - `evidence.py` declines retries where the stem is clearly quiet. Its `instrumental_song`
     keeps clearly instrumental songs from being transcribed at all.
3. **Timing repair** (`transcription`): drafts whose timestamps are compressed (`timing.py`) are
   realigned with the karaoke aligner, loaded once, on the draft's own audio span
   (`provided_audio=True`).

On Modal the same phases run remotely, with results streaming back in order.

Afterwards `refinement.py` is the optional External AI second pass. It sees the whole song and
returns corrected text per line, keeping MOSS's line count, order and timing. A line that comes
back empty or invalid keeps its transcribed text.

## What it stores

Nothing directly. The lyrics stage stores the transcript in `lyrics`, with `ai_generated` and
the transcription record in its provenance, and later the `refinement` record. A transcript is
made only for songs without lyrics, or when a user forces transcription.

## Configuration

`TranscriptionConfig` (`config.py`), prefix `ECHORA_TRANSCRIPTION_`:

- `moss_model_id` and `moss_revision` (older names `MOSS_MODEL_ID`, `MOSS_REVISION`). Both must
  be set, the revision as a 40-character commit, or a `ConfigurationError` is raised.
- With neither set, AI lyrics are unavailable; there is no implicit upstream fallback.
- The snapshot must already be downloaded (`python -m worker.models.download`).

Refinement uses the External AI settings and its `transcription_refinement` switch. See
[runtime settings](../../../docs/runtime-settings.md).

## Failures

- `WindowDecodeError` marks a window that could not be decoded; unresolved regions are reported
  in the transcript's record instead of failing the song.
- `RefinementError` and every other refinement failure keep the transcript as it was.
- A song whose transcription or timing repair raises fails in the lyrics stage, and the next run
  retries it.
- Failed-window text is written only to container-local diagnostic files, not to the database.

## See also

- [Song transcription](../../../docs/song-transcription.md)
- [Batch phases](../../../docs/batch-phases.md)
- [`vendor/moss`](vendor/moss/README.md): the MOSS processor code and its licence.
