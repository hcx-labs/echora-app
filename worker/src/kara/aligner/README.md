# Karaoke aligner

Times every word of a song's lyrics from the audio alone. The lyrics say which words are sung and
in which line; the timings come only from the sound, never from source timestamps.

It began as [FA-Kara](https://github.com/moriwx/FA-Kara) (MIT, see `LICENSE`, commit `168ca5f`)
and has since been rewritten around the Echora Kara models
([hcxlabs/echora-kara-700m-muq](https://huggingface.co/hcxlabs/echora-kara-700m-muq), trained in
[echora-kara](https://github.com/hcx-labs/echora-kara)). That commit id is still part of every
stored alignment's revision (`FA_KARA_REVISION`), so stored alignments stay current.

## How a song is aligned

1. **Labels** (`normalize.py`, `label_rules.py`, `romanization.py`, `indic.py`, `korean.py`).
   Each lyric line gets a language (`label_rules.line_language`) and becomes acoustic tokens in
   the model's Latin alphabet: kana and kanji through readings, Hangul by pronunciation
   (같이 → ga+chi), Han as pinyin or Jyutping, Indic and Arabic scripts romanised, English by its
   sung pronunciation. The label rules (`v3-sung` by default) must be the ones the model was
   trained with.
2. **Emissions** (`decoder.py`, `muq_fusion.py`). The checkpoint (wav2vec2 / w2v-BERT CTC with a
   frozen MuQ branch) runs over the full mix in overlapping chunks, stitched into one lattice on
   the true frame clock. A backing-vocal head and a text head come with the checkpoint.
3. **Decoding** (`ctc.py`, `text_aligner.py`). One global Viterbi pass over the whole song. Each
   token has a minimum duration scaled by its line's pace. A frame where the vocals are quiet costs
   extra, and letters of one token have no blanks between them, as in training. The text head adds
   its evidence (`FA_KARA_TEXT_BONUS`).
4. **Backing vocals.** A line that is one bracketed span is decoded on the backing-vocal head and
   may overlap the lead. Inline brackets stay on the lead path for order, then are placed on the
   backing head inside their line's window.
5. **Document** (`align.py`). Every source line with its language, its timed tokens (integer
   milliseconds, CTC score, voice) and diagnostics. `schema_version` 1 is what the karaoke
   pipeline validates and stores.

## Running it

The karaoke pipeline (`worker/src/kara/pipeline.py`) starts `python -m worker.kara.aligner.worker`
once and sends it one `AlignRequest` (`request.py`) per song as a JSON line. The worker keeps the
models loaded between songs. The pipeline stops it before vocal separation or another model needs
the GPU, and kills it on cancellation.

`decoder.py` still reads a few decoder experiment switches from the environment (`FA_KARA_*`;
everything is off unless set). Production sets two of them: the text bonus and the in-line gap
cost, from `KaraConfig`.
