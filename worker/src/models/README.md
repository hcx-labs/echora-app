# Models (`worker.models`)

Downloads the pinned model files analysis needs, and keeps the stored Hugging Face token for gated
downloads. It is no stage; it runs as `python -m worker.models.download` (on a machine with
network access) and as Modal's model preparation (`analysis_downloads()`).

## Flow

- `required_models()` lists the pinned snapshots (repository, commit, and whether its local
  `main` ref must point at that commit):
  - the embedding and voice models (`MODELS`);
  - the karaoke aligner checkpoint (`KaraConfig`);
  - the Roformer separator;
  - the transcription model, when one is configured.
- `main()` downloads the recording-search model, the motion artwork files (when wanted), every
  snapshot and the Essentia voice classifiers (checked by SHA-256). It then prunes superseded
  snapshots of managed repositories from the Hugging Face cache, leaving other repositories
  alone. `--prune-only` only prunes; `--motion-artwork` also fetches the motion artwork models.
- `model_manifest()` is a digest of every pinned model. Modal compares it with the manifest on
  its model volume to decide whether storage must be prepared again.
- `analysis_downloads()` gives the same downloads as labelled steps, for progress.
- `huggingface_token.py` stores the token from Settings → Analysis models, encrypted with
  `CREDENTIAL_ENCRYPTION_KEY`. `token_or_none()` returns it when `HF_TOKEN` is not set; `save()`
  checks it with Hugging Face first.

## What it stores

The Hugging Face cache and the Essentia directory on disk; the encrypted token in
`analysis_settings`. A pinned revision change (in `core/src/revisions.py`, `KaraConfig`,
`EmbeddingConfig` or the transcription settings) changes the manifest and the next download.

## Configuration

`ModelsConfig` (`ECHORA_MODELS_`): `essentia_dir`. Downloads honour `HF_HOME` and `HF_TOKEN`.
See [docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

- A download or checksum failure raises and stops the command; run it again.
- Hugging Face refusing a token raises `TokenRejected` (`worker.models.errors`) when it is saved.
- A stored token that cannot be read counts as none (logged by exception type only).

## See also

- [docs/modal-compute.md](../../../docs/modal-compute.md)
- [docs/runtime-settings.md](../../../docs/runtime-settings.md)
