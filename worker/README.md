# Worker (`worker`)

The analysis and scheduled workers: they claim jobs from the queue (`core.jobs.queue`), run the
analysis a job asks for over its songs, and store the results. Code lives in `worker/src` and is
imported as `worker`; shared models, configuration and storage live in `core`.

## How a job runs

1. **`worker.executor`** (`python -m worker.executor analysis|scheduled`) polls for a job of its
   worker type, then runs each claim in its own spawned process, so model memory goes away with
   the claim. The supervisor heartbeats the lease and stops the process when the claim is lost
   or the worker shuts down. A summary with per-song failures (`failed` or `*_failed` above 0)
   is retried until the job's attempts run out, except failures in `translations`; after that
   it completes as `partial`.
2. **`worker.operations`** maps a job kind (`navidrome_sync`, `import`, `lyrics_backfill`,
   `karaoke_backfill`, `voice_backfill`, `recordings_backfill`, `hum_corpus`, `audio_profiles`,
   `semantic_fusion_build`) to an `Operation`: its stages, and whether it reads a library and
   expands into batches. A parent job plans its songs and queues `analysis_batch` children; a
   batch runs every stage's `prepare` hook, opens a Modal session when it computes remotely,
   then runs its stages in order and merges their summaries.
3. **`worker.stages`** defines each `Stage`: a summary key and a function of the `Batch`. Most
   build one processor and run it (`processor_stage`).
4. **Processors** (`worker.processing`) do the work: plan the songs that need the stage, compute,
   store each result, report progress. See [processing/README.md](src/processing/README.md).

The scheduled worker refreshes curations instead (`worker.curation`).

## Domains

| Package | What it does |
| --- | --- |
| [library](src/library/README.md) | Navidrome ingest: track identity, waveforms, visual features, audio embeddings, melody, fingerprints, descriptors |
| [lyrics](src/lyrics/README.md) | Lyrics retrieval or transcription, refinement and lyric embeddings |
| [transcription](src/transcription/README.md) | AI lyrics from audio (MOSS), timing repair, External AI refinement |
| [kara](src/kara/README.md) | Karaoke word timing; the aligner itself is `kara/aligner` |
| [translation](src/translation/README.md) | Lyric translation |
| [themes](src/themes/README.md) | Lyric theme descriptions |
| [voice](src/voice/README.md) | Voice and vocal-gender classification |
| [melody](src/melody/README.md) | Melody contours and hum corpora |
| [recordings](src/recordings/README.md) | Recording fingerprints and recording search |
| [profiles](src/profiles/README.md) | Audio profiles from stored embedding windows |
| [fusion](src/fusion/README.md) | Semantic fusion vectors (lyrics + audio) |
| [artwork](src/artwork/README.md) | Motion artwork (ComfyUI) |
| [curation](src/curation/README.md) | Scheduled curation refreshes |
| [media](src/media/README.md) | Audio decoding, the prepared-audio cache, Roformer separation |
| [planning](src/planning/README.md) | Which songs need which analysis |
| [processing](src/processing/README.md) | The Processor framework |
| [models](src/models/README.md) | Downloading the pinned models |
| [remote](src/remote/README.md) | Model computation on Modal |

## See also

- [docs/jobs.md](../docs/jobs.md): the job queue.
- [docs/batch-phases.md](../docs/batch-phases.md): how a batch orders its phases and models.
- [docs/runtime-settings.md](../docs/runtime-settings.md): settings and environment variables.
