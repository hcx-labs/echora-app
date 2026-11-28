# Karaoke (`worker.kara`)

Word- and syllable-level karaoke timing for a song's lyrics, aligned to its full mix with the
Echora Kara aligner. It runs as the `karaoke` stage (`worker/src/stages.py`) in `navidrome_sync`,
`import`, `lyrics_backfill` and `karaoke_backfill`, always after the lyrics stage, so it aligns the
lyrics that stage keeps current. The aligner itself lives in [`aligner/`](aligner/README.md);
`furigana.py` builds Japanese readings from the aligner's own units.

## Flow

`KaraokeProcessor` (`pipeline.py`), progress phases `preprocess`, `models` and `karaoke`:

- **plan**: `plan_karaoke` (`worker/src/planning/audio.py`) selects the batch's songs with lyrics
  and no current karaoke variant, and only while the `karaoke` feature is on.
- **open**: takes the worker's karaoke lock (one batch aligns at a time), opens the database and
  Navidrome.
- **process**: songs whose lyrics came with word timing (TTML or enhanced LRC from Navidrome) are
  stored as they are (`supplied`). The rest are fetched first, every one of them (`preprocess`;
  on Modal each song is uploaded once), and their vocals separated into the prepared-audio cache
  (`stem_reading`); then the aligner loads once (`models`) and aligns them all. The aligner hears
  the full mix and the separated vocals, and adds what it reads on the vocals to what it reads on
  the mix: instruments mask quiet singing, the separator removes some of it, and the mix stays
  the reference (a letter the mix hears is never lost).
  The aligner runs in a resident worker process (`python -m worker.kara.aligner.worker`) that
  takes one `AlignRequest` per song; on Modal every song aligns in one call.
- **persist**: writes the alignment as the song's karaoke variant.
- **close**: stops the aligner process and releases the lock.

Summary `karaoke`: `total`, `aligned`, `supplied` (when any), `failed`.

## What it stores

- `karaoke_lyrics_variants` (one unbounded variant per song): lines, model, model revision and
  provenance, including `pipeline_revision`.
- `lyrics.karaoke_*`: the same lines, mirrored for the player. ASS and LRC are no longer written.

### Stored readings

With `ECHORA_KARA_READINGS_DIR` set, every full alignment also keeps the song's *model reading*:
what the decoder reads from the model (`decoder.ReadingRecorder`), i.e. the stem's and the mix's
letter log-probabilities, the backing head, edge logits, the frame clock and the frame embeddings
the text head reads (float16; on 16 songs every syllable decoded the same as from float32). About
20 MB a song, as `<dir>/<id[:2]>/<track id>-<recipe digest>.pt`, also from Modal.

A song to realign whose reading is of the current `reading_recipe()` (reading revision,
checkpoint and the audio it hears, separator, MuQ branch, audio speed, text head) is decoded from
it in the worker (`decoder.StoredReading`): no separation, no model, no Modal, a few seconds a
song. So a decoder change (`KARAOKE_PIPELINE_REVISION`) or new lyrics re-decode the library from
the readings; a model change reads every song again and replaces its reading. Readings are not
a cache with a budget: one file per song, replaced when its recipe changes.

A song is redone when its stored `pipeline_revision` differs from `KARAOKE_PIPELINE_REVISION`
(`core/src/revisions.py`) or its model revision differs from the current one. The model revision
is `<FA_KARA_REVISION>:<checkpoint revision>:mix+stem` (`:mix` with `stem_reading` off); for a local checkpoint
(`ECHORA_KARA_MODEL_PATH`) it is a digest of the weights. Supplied timing is current while its
`source_digest` matches the lyrics' `word_timing_digest`.

## Configuration

`KaraConfig` (`config.py`), prefix `ECHORA_KARA_` (older `FA_KARA_*` names are still read). It
holds the checkpoint (`model_id`, `revision` or `model_path`), the decoder values the model was
evaluated with (duration floors, frame clock, backing-vocal decoding, `text_bonus`,
`in_line_gap_cost`, `fusion`, `label_rules`) and the worker process timeouts. Change decoder
values together, and only with an evaluation. Every setting is listed in
[runtime settings](../../../docs/runtime-settings.md).

`repeat_template` (`ECHORA_KARA_REPEAT_TEMPLATE`, off) is an experiment awaiting a benchmark. A
line the model hears clearly and that recurs (at least three times) anchors a block. The block's
lines before each occurrence are compared by their offset from it, and an occurrence that
disagrees with the majority takes the majority's timing (`aligner/repeats.py`). It runs on the
finished alignment document, so a replayed reading applies it without the model.

## Failures

- A song whose audio cannot be fetched fails with `InputUnavailableError`.
- An alignment that raises fails with `ProcessingError("Alignment failed")`, with the exception
  type as its stage.
- Failed songs are counted in `failed`. They get no variant, so the next sync or backfill plans
  them again.
- The worker process is replaced when the model revision changes, or when the text head becomes
  needed or unneeded.

## See also

- [Aligner](aligner/README.md)
- [Batch phases](../../../docs/batch-phases.md)
- [Modal compute](../../../docs/modal-compute.md)
