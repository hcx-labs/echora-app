# Analysis on Modal

Echora can run a sync's machine-learning work on [Modal](https://modal.com) cloud GPUs instead of the server's own hardware. The sync is still an ordinary job, claimed and run by a local analysis worker; only model computation moves. A server without an NVIDIA GPU can then offer karaoke timing, AI lyrics generation and melody search.

## Setting it up

1. In Modal, create an API token (Settings → API Tokens). A payment method is required for GPU functions; Modal's monthly free credit still applies.
2. In Echora, open **Settings → External processing** (administrators only). Enable Modal, enter the token ID and secret, and choose a GPU type. Saving verifies the token and shows its workspace. The secret is encrypted with `CREDENTIAL_ENCRYPTION_KEY`, like other stored credentials.
3. Optionally press **Prepare Modal**. A background job deploys Echora's analysis app to the workspace and downloads its models (about 20 GB) into a Modal Volume, with progress on the page. The first Modal sync does the same if this step is skipped.
4. In **Settings → Analysis models**, choose which optional features run in syncs on this server and which in syncs on Modal.
5. When starting a sync, choose **Where to run**: this server or Modal. The choice is preselected from External processing settings.

By default only administrators can run syncs on Modal; **Let all users choose Modal** extends this to everyone, at the instance's expense. Jobs not started from the sync page (backfills, imports) run in the preselected location.

## Features per location

Every sync runs in exactly one location, and does the optional features switched on there:

| Feature | Switch |
| --- | --- |
| AI lyrics generation (Roformer vocals, MOSS) | This server / Modal |
| Karaoke timing (Roformer vocals, FA-Kara) | This server / Modal |
| Query by humming (Roformer stems, Melodia) | This server / Modal |
| Motion artwork (LTX-2.5 in ComfyUI) | This server / Modal |
| Audio embeddings (MuQ-MuLan, MERT) | Always |
| Lyrics embeddings (BGE-M3) | Always |
| Voice detection (Essentia classifiers) | Always |

Motion artwork on Modal downloads the gated LTX-2.5 model (about 44 GB, once) with the Hugging Face token saved in **Settings → Analysis models → Model downloads**, the same token local downloads use. Modal receives it for that download only. Loops render in a separate GPU class with a private ComfyUI, and finished videos come back through the transfer Volume, then are stored and recorded on this server.

A feature switched on only for Modal is skipped by syncs on this server and left pending; the next Modal sync processes it, and the reverse holds too. Sync planning (what counts as missing) follows the sync's location.

CPU work stays on the server in both locations: waveforms, fingerprints, audio descriptors, visual features, audio profiles and translations (which use External AI).

## What runs where

Modal cannot reach the database, Navidrome or lyrics providers, so:

| Local analysis worker | Modal |
| --- | --- |
| Claims the job, plans the batch, reports progress | Runs model inference only |
| Fetches audio from Navidrome and lyrics from providers | Receives audio by content hash, and lyrics text |
| Writes every result, with provenance, to PostgreSQL | Returns results; keeps no state but caches |

## Flow of a Modal sync

1. The sync job records `compute: "modal"`, and every batch inherits it.
2. Before batching, the job prepares Modal: under a database lock, it compares the deployment with this worker and fixes whatever is stale (below).
3. Each batch checks the deployment again (one small call, about 2 seconds) before uploading anything, so a batch never runs on stale code or models.
4. The worker uploads each track's audio once per batch to the `echora-audio` Volume, named by its SHA-256.
5. Each stage keeps its own order, planning, failure accounting and progress, and runs in the same model phases as locally ([batch-phases.md](batch-phases.md)). Where it would run a model, it sends all of its tracks in one call; results stream back in order and are stored as they arrive. Transcription streams its progress and window diagnostics too.
6. When the batch ends, its audio is deleted from Modal.

A track that fails on Modal counts as one failed track, like a local failure, and is retried by the existing batch retry policy. If Modal cannot be prepared, the batch fails with the reason, and the External processing page shows it.

## Keeping Modal current

The deployment is checked against three things, and only what differs is redone:

- **Code.** A digest of the worker's analysis package and vendored aligner. The worker deploys its own source on top of the base image, so remote code is exactly the worker's. An Echora upgrade redeploys.
- **Image and GPU.** Release images record the GPU analysis image of the same version (`ECHORA_MODAL_IMAGE`, set at build time), so remote dependencies match. Changing the GPU type redeploys.
- **Models.** A manifest digest of every pinned model the worker's settings require. A changed model setting (for example a new FA-Kara revision or enabling MOSS) downloads only the missing snapshots, on a CPU container so downloads do not bill GPU time, and removes superseded ones.

Every compute call also carries the worker's model settings (model IDs, revisions, aligner options), and the remote side computes with exactly those, so results match what the worker would compute locally. Measured on the same song, Modal reproduces locally stored embeddings at cosine 1.00000.

## The Modal app

`services/analysis/src/echora_analysis/modal_app.py` defines the app (`echora-analysis`): a CPU `status` function, a CPU `prepare_models` generator, and a GPU `Analysis` class with one method per stage. The GPU container keeps one model family resident at a time, as local processing does, scales down after five idle minutes and runs at most one container. Echora deploys it through the Modal CLI with the stored token only, isolated from any Modal profile on the machine.

To deploy by hand, from `services/analysis` with a Modal token configured:

```sh
ECHORA_MODAL_IMAGE=ghcr.io/hackslashx/echora-analysis-gpu:<version> \
  modal deploy -m worker.remote.modal_app
```

## Interactive features

Text queries (sound tags, lyrical themes, concept search) encode short text with MuQ-MuLan and BGE-M3 at request time on the server. Without a GPU these run on the CPU, which is slower but acceptable for short text.

## Privacy and cost

A Modal sync sends source audio and lyrics to the Modal workspace. It is never the default unless an administrator preselects it. GPU time is billed per second by Modal while containers run, including model loading; containers stop after five idle minutes.
