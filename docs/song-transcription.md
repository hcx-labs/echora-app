# AI lyric transcription

The analysis package can recover lyrics with shared Mel-Band Roformer vocals at overlap 2 and the published Echora MOSS fine-tune. This is an optional fallback, not a replacement for supplied lyrics.

## Configuration

Compose defaults to the published checkpoint:

```dotenv
MOSS_MODEL_ID=hcxlabs/echora-kiki
MOSS_REVISION=b35a2d893c8c287c6f77da7c2931ac17df0b90df
ECHORA_BATCH_SIZE=128
```

Set both MOSS variables explicitly to empty strings to disable transcription. Outside Compose, no model is enabled unless configured.

To select another checkpoint, set the HF repository ID and its immutable 40-character commit revision. Partial configuration fails explicitly. There is no automatic fallback to base MOSS or an experiment-directory checkpoint.

The shared `echora_analysis.download_models` command downloads the configured MOSS snapshot and pinned Roformer checkpoint. Run it with the same environment and model volumes as the worker, with HF offline mode disabled for provisioning. Inference loads MOSS from local cache only. The HF snapshot must include the fine-tuned model, tokenizer, processor configuration and required custom modeling code. Custom code executes from the reviewed pinned snapshot.

The processor is vendored from the upstream repository with its Apache license. The model weights are not included in the application repository. Review the published weights' license and training-data redistribution permissions before release.

## Processing order

Library import/sync and lyrics backfill retrieve source lyrics first. For tracks without text and not reported instrumental, the pipeline transcribes vocals before lyric embedding and karaoke alignment. Karaoke backfill now runs the lyrics stage first too.

- Resolve supplied and existing lyrics for the batch before preparing actual transcription fallbacks.
- The `preprocess` phase decodes source audio to 44.1 kHz stereo and stores Roformer vocals in the shared local prepared-audio cache. Compatible downstream resampling is cached too.
- Complete separation before loading MOSS so the models do not compete for GPU memory. Karaoke reuses these vocals instead of separating again. See [shared audio preprocessing](audio-preprocessing.md) for cache limits and invalidation.
- Process 60-second windows with 12-second overlaps through the entire song.
- Apply segment-local lexical repetition protection. Preserve Sxx and MULTI tags without claiming real speaker identities.
- Merge by segment midpoint ownership. No global text deduplication that would erase repeated choruses.
- Retry incomplete formatting, invalid intervals and token-limit outputs as shorter overlapping windows within the shared song budget. Stop identical stalled-timestamp loops early.
- After bounded retries, omit unresolved windows and retain usable regions. Store `partial=true` and `unresolved_windows` in transcription provenance rather than claim complete coverage. If no usable lines remain, fail the track.
- Save failed-window raw output in private container-local `/tmp/echora-transcription-*.json` files; paths are logged. These diagnostic files disappear when the container is replaced.
- Store successful text as `source=transcribed`, with millisecond source lines, `synced=true`, `ai_generated=true`, model revision and decoding provenance.

Provider misses do not erase existing lyrics. AI candidates cannot overwrite existing nonempty lyrics. The karaoke stage merges its own provenance and retains the AI flag. The fullscreen player shows a notice in both synced and karaoke views, on desktop and mobile.

The job batch default is 128 tracks, not a GPU inference batch of 128. Inference remains sequential and cancellation is checked between windows and phases. Failed tracks remain eligible for retry. Transcription is disabled when the model configuration is empty.


## External AI refinement

When External AI is enabled, each new AI transcript gets a second pass during the same sync, after transcription releases its models and before lyrics are embedded, translated and aligned for karaoke. The configured model reads the whole song (title, artist, detected languages and the timed lines) and corrects mis-heard words so each line makes sense on its own and within the song. Line count, order and timing stay the transcript's; a line the model leaves empty or returns invalid keeps its transcribed text, and any failure keeps the whole transcript.

The switch and the prompt are under Settings → External AI → Transcription refinement. It is on by default and acts only while External AI is enabled; the response format is appended in code, so an edited prompt cannot break it. The lyrics record keeps the original transcript lines, the model's song summary and the prompt revision under `transcription.refinement`.

## Limitations

Window boundary predictions may disagree, and the midpoint merge can miss or duplicate a line. Valid formatting is not proof of lyric accuracy. Partial recovery does not detect every hallucination: MOSS can emit plausible, well-formatted words during instrumentals. Unresolved-window ranges describe failed attempts and may overlap neighboring accepted windows. MULTI semantics are not verified. Vocal separation can introduce artifacts. Instrumentals incorrectly labeled as missing may produce hallucinations. No confidence score or accuracy guarantee is presented to users.

Provision the pinned snapshot through the shared downloader before starting workers with transcription enabled. An end-to-end worker/player smoke test remains required before release; unit and frontend build checks do not replace it.

## Processing control and retry limits

Settings → Model processing → Generate missing lyrics with AI is an administrator-only, application-wide switch. The database defaults to off. A configured model does not enable transcription by itself. Planning and each track check the current setting. Disabling it lets the current track finish but prevents subsequent tracks from starting transcription. Source lyric retrieval and existing lyrics are unaffected.

The production Roformer timing-v7 pipeline preserves the timing-v6 recovery policy and attempts every original window before recovery. Failed windows may receive one overlapping split, with no recursive retries. Each song shares four additional decode calls and 4,096 generated retry tokens, capped at 1,024 tokens per call. Failed calls count toward both limits. Generation enforces the per-call token allowance; there is no wall-clock timeout. Missing or conflicting vocal evidence preserves eligibility for this bounded retry pass. Unresolved regions and actual retry usage are saved in provenance. Progress reports the current window and retry call.

Local deployments can explicitly enable the switch after applying migration 0040. Never change the migration default to enable a particular instance.

## Repair compressed timestamps before merging

Generation completes before timing reconciliation. Normally MOSS timestamps remain unchanged. A candidate qualifies for repair only when it has at least four lines and 24 whitespace-delimited words, at least half its lines are distinct, its output spans at most 35% of its window, and its average density is at least six words per second. These initial thresholds detect suspicious compression, not lyric accuracy. They do not reliably cover languages without word spacing.

At most two candidates per song receive forced alignment before retry-half or outer-window ownership filtering. The repair reuses the existing vocal stem, sends float WAV rather than clipping to integer FLAC, disables additional vocal separation, and moves MOSS to CPU before loading the aligner. It preserves the original text and speaker labels and requires the same ordered lines with positive, bounded, monotonic timestamps. It never globally deletes repeated phrases. Ordinary alignment failures leave the original candidates in place with an unresolved timing diagnostic; cancellation and memory errors propagate.

`transcription.timing_repairs` records attempted repairs and their model revision. Reaching the timing-repair limit is recorded in window diagnostics. Aligning every window was rejected after regression testing; timing-v6 leaves ordinary windows unchanged. A four-song evaluation restored the reported Almost Forgot passage while preserving the other three transcripts exactly. Only two songs had independent text references, so this is not a general accuracy guarantee.
