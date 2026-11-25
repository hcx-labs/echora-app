# Remote (`worker.remote`)

Model computation on Modal. A batch whose job chose `compute: "modal"` opens a session, and
stages ask the session to run a model where they would otherwise run it locally. Each stage
keeps its own planning, storage and progress. Admins enable it in Settings → External
processing.

## Flow

- `settings.py`:
  - holds the instance-wide settings: the Modal token (encrypted with
    `CREDENTIAL_ENCRYPTION_KEY`), whether users may use it, and the default compute;
  - `compute_for()` decides where a new job computes;
  - `prepare()` deploys the app and prepares its models, under a database lock, with progress
    in the `modal` phase. A sync parent runs it once before queueing its batches; the
    `modal_setup` job runs it from the settings page;
  - `open_session()` gives the batch's `ModalSession`.
- `session.py`: `session(compute)` sets the batch's `ModalSession` (`current()`) for the
  operations runner; stages read it as `context.remote`.
  - `ensure_ready()` checks that the deployment runs this worker's code (`code_identity()`) on the
    configured image and GPU, with the model set the settings require. It redeploys or prepares
    models first when anything is stale.
  - Source audio is uploaded once per batch, named by its SHA-256, and removed when the session
    closes.
  - Batched calls (`embed_audio`, `melody`, `prepare_vocals`, `voice`, `karaoke`,
    `embed_lyrics`, `transcribe_*`, `motion_artwork`) stream one result per input, in order;
    an input that failed yields its exception.
- `compute.py`: every config class with `compute()` fields travels with each call
  (`compute_settings()`), and the container adopts it (`adopt_compute_settings()`), so remote
  results match local ones.
- `modal_app.py`: the deployed app. Its image is Echora's GPU worker image with the deploying
  worker's source on top; model weights live on a Modal volume, with a manifest of the pinned
  model set.

## What it stores

`external_processing_settings` (token, switches, deployment record). On Modal: the models volume
and the per-batch audio volume. Nothing is computed remotely that a local run would not store
the same way.

## Configuration

`RemoteConfig` (`ECHORA_MODAL_`): `image` and `gpu`. Release images set their own image; the
worker records the GPU when it deploys. The token and switches are database settings, not
environment variables. See [docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

- A deployment that still does not match after deploying, or incomplete model storage, raises,
  and the batch fails before any song is sent.
- A user asking for Modal without access gets `PermissionError`.
- Per-input failures come back as exceptions in the result stream, and each stage counts them
  for its song.
- Modal CLI output is scrubbed of the token before it is raised.
- Cleanup failures are logged and ignored.

## See also

- [docs/modal-compute.md](../../../docs/modal-compute.md)
- [docs/batch-phases.md](../../../docs/batch-phases.md)
