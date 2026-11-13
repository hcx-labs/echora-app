# Batch phases

Analysis works on batches of songs. Every pipeline runs as a sequence of **phases**, and each phase holds at most one large model:

1. Load the phase's model once (only if a song in the batch needs it).
2. Run every selected song of the batch through it.
3. Release it before the next phase loads its model.

Per-song failures, progress and cancellation work the same in every phase. A song that fails a phase is recorded as failed and skips the later phases of that pipeline; the rest of the batch continues.

The same phases run on this server and on Modal ([modal-compute.md](modal-compute.md)). On Modal each phase is one batched call, and the GPU container keeps the phase's model loaded between songs.

## Phases per pipeline

| Pipeline | Before | Main | After |
| --- | --- | --- | --- |
| Audio embeddings | decode (CPU) | MuQ-MuLan, then MERT | — |
| Melody | Roformer vocal separation | Melodia contours (CPU) | — |
| AI lyrics | Roformer vocal separation | MOSS generation | timing repair with the FA-Kara aligner |
| Lyrics embeddings | — | BGE-M3 | — |
| Karaoke | Roformer vocal separation | FA-Kara alignment | lead-in guard (CPU) |
| Voice | decode (CPU) | voice classifiers | — |
| Motion artwork | prompts (External AI, at the start of the batch; or Gemma) | LTX prompt encoding | LTX video |

Karaoke skips all three phases for a song whose lyrics arrive from Navidrome with word timing for every line (a TTML file, or LRC with word cues): that timing is stored as the karaoke as is, lead singers on the line and `ttm:role="x-bg"` agents as the second voice, with the lyrics' `offset` applied. It is not AI-generated, so players show no AI disclosure for it. Partial word timing is ignored and the song is aligned. Navidrome does not say where a TTML file came from: a local stack syncing a Navidrome that another Echora's plugin serves would take that Echora's AI karaoke as supplied timing.

## Rules

- **Only selected songs.** Each pipeline's planner selects the songs that need it (missing or stale results, feature switched on for the batch's location). A phase never processes, separates or loads a model for anything else. When no song is selected, the phase loads nothing.
- **Shared preprocessing.** Separated vocals and decoded audio are cached by the source audio's content. The first phase that needs a song's vocals separates them; any later phase in the batch reads the cache. A song is separated at most once per batch (on Modal: per GPU container, which stays up between phases).
- **Persist as soon as possible.** A song's result is stored when its last needed phase finishes. Lyrics that need no timing repair are stored right after generation; only the few that do wait for the repair phase.
- **Nothing waits on a GPU.** Work that needs no GPU and could leave one idle runs before the GPU phases start. Motion artwork prompts from External AI are written at the start of each batch, from Navidrome's data, so neither this server's GPU nor a Modal container waits on them. A brand-new song that is transcribed later in the batch is therefore prompted without lyrics.

## Building blocks

- `roformer.separation_phase()` holds one Roformer for a phase; `roformer.separate_vocals` uses it. Outside a phase each call loads its own model, as before.
- `SongTranscriber` loads MOSS with `open()`/`loaded()`, `generate()` produces a draft per song, and `finish()` completes it (timing repair needs the aligner, not MOSS). `transcribe()` still does all three for a single song.
- On Modal, the `Analysis` class keeps one resident model (`_resident`) and releases it when a phase asks for a different one; `prepare_vocals` is the separation phase, called with exactly the artifacts the asking pipeline reads.

New pipelines should follow the same shape: plan the songs, then one phase per model, each loading once for the batch.
