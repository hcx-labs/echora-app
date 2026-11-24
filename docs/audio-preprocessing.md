# Shared audio preprocessing

Analysis batches prepare the audio formats required by outstanding work before loading downstream models. Preparation appears as the `preprocess` job phase. This is a stage within the existing leased batch, not a new independently scheduled job.

Planning remains incremental. Imports resolve source-byte identity before checking missing audio analyses. Lyrics retrieval checks supplied and stored text before deciding whether transcription needs vocals. Karaoke eligibility follows the stored synchronized lyrics. Completed analyses do not trigger preparation merely because a song belongs to a batch.

## Vocal-only tasks

Transcription, karaoke and melody extraction use one Mel-Band Roformer recipe:

- Checkpoint `KimberleyJSN/melbandroformer`, revision `ac9b0614ab3cd7f77219e18ba494dfd93956c348`, file `MelBandRoformer.ckpt`.
- Source architecture/configuration at commit `25f44ffb55ee3c301281bba21b2d6d311cb69ae2`.
- 44.1 kHz stereo input, eight-second chunks, overlap 2. The stride is four seconds.
- Float32 output without clipping or independent normalization. GPU inference uses mixed precision.
- Shared mono vocal resampling for downstream models. Timestamp repair explicitly bypasses separation because it already receives vocals.

The separator releases its model before transcription or alignment begins. It loads only the pinned local checkpoint. The production transcriber remains the configured Echora MOSS fine-tune, not the experimental MOSS-Music 8B model.

## Reuse and exclusions

| Work | Shared preparation |
| --- | --- |
| MuQ and MERT | Identical 24 kHz mono decoding |
| Waveform visualization | 24 kHz stereo decoding, kept distinct from mono |
| Descriptors and melody separation input | 44.1 kHz stereo decoding |
| Full-mix melody contour | 44.1 kHz mono decoding |
| Voice-presence classification | 16 kHz mono original mix, never a vocal stem |
| Transcription and karaoke | Roformer vocals and compatible mono resampling |
| Vocal and accompaniment melody contours | Shared Roformer vocals plus cached mix-minus-vocals, downmixed to 44.1 kHz mono |

Melody extraction only needs vocals and combined accompaniment, not separate drums, bass or other instruments. It now reuses Roformer vocals and subtracts the stereo vocal estimate from the same decoded stereo mix before downmixing the residual. This replaces the active Demucs implementation and dependency. The downloader no longer fetches or deletes Demucs weights; existing experiment files remain untouched. Legacy vendored FA-Kara's optional Demucs CLI is not used by the service and is no longer supported by its dependency set.

Chromaprint still receives the original encoded audio because changing its decoding path could alter fingerprint compatibility. Audio profiles consume stored embeddings and need no audio preparation. Window slicing is cheap and remains in each embedding consumer rather than storing duplicate window arrays.

Karaoke's original-mix reference is downmixed from the same cached 44.1 kHz stereo input as the vocal estimate, then resampled with the identical soxr recipe. This keeps both 16 kHz arrays on exactly the same sample grid; independently decoding the reference straight to 16 kHz can round its length differently and fail alignment. The reference remains the original mix, not separated audio.

The existing authenticated source-download cache is still private to one batch. It is not made persistent across users or jobs, because a URL/external ID alone cannot prove that remote bytes are unchanged. Prepared audio becomes reusable only after the caller obtains authorized source bytes and hashes them.

## Local storage and lifecycle

Compose stores prepared arrays and manifests under `data/preprocessed`, mounted as `/data/preprocessed`. Workers need write access. API image startup creates this directory for service UID 1001 before dependent workers start. For external deployments, create and assign the configured directory yourself.

Configuration:

```dotenv
ECHORA_PREPROCESS_DIR=/data/preprocessed
ECHORA_PREPROCESS_MAX_BYTES=21474836480
ECHORA_PREPROCESS_TTL_SECONDS=604800
```

The defaults are 20 GiB and seven days since last use. Eviction is LRU/TTL-based when publishing artifacts. Active locked entries are skipped, so temporary usage may exceed the target. Entries larger than the budget are usable but not retained. Zero budget disables retention. Stable lock files are intentionally not deleted, avoiding lock-inode races.

Size the budget for the whole batch, not just one track: prerequisite preparation runs ahead of its consumers, and completed entries are not pinned until later stages finish. A 128-track melody batch can require roughly 40 GiB (duration-dependent), exceeding the 20 GiB default and evicting stems before reuse. On hosts with enough free space, `ECHORA_PREPROCESS_MAX_BYTES=68719476736` gives a 64 GiB budget; otherwise lower `ECHORA_BATCH_SIZE` before creating a new sync. Changing batch size does not resize already-queued batches. Keep sufficient filesystem headroom for other data and temporary publication files.

Keys include the SHA-256 of original source bytes, format and preparation revision. Vocal keys include the pinned model, model configuration, chunk size, overlap and implementation revision. Resampled vocal keys include sample rate and resampler version. Changing a recipe produces a new key. The decode recipe revision must change when deliberately changing decoding semantics.

Cross-process file locks prevent simultaneous production of the same cached artifact. Staging files publish through atomic replacement; a checksum-verified manifest marks a complete result. Corrupt, expired or interrupted entries rebuild. Cancellation propagates through lock waits, separator chunks and publication. A retry can reuse completed preparation without treating unfinished inference as complete.

The cache is private service data, not a playback API or authorization mechanism. Do not expose it as a static web directory. Removing cache entries only costs recomputation; do not delete lock files while workers run. Stop workers before clearing the whole directory. Source credentials and URLs are not written to manifests.

## Deployment and validation

Provision model snapshots with the existing `worker.models.download` command in online mode before starting offline workers. Docker startup only prunes managed snapshots; it does not fetch missing models. Legacy `FA_KARA_DEMUCS_MODEL` and `FA_KARA_VOCAL_SEPARATION` no longer select the service's vocal-only input.

New karaoke and melody recipe identities make old results eligible for regeneration on the next applicable job. Hum search selects only the current Roformer melody revision. Refresh the existing index with a melody-index job or an entire-library sync with hum processing enabled before searching. Old contours remain stored for provenance; no bulk deletion or rebuild happens at deployment. Failed separation does not publish a full-mix-only result as a completed new recipe. Sources with insufficient pitch evidence can still be omitted after successful separation.

Existing nonempty lyrics remain protected, including prior generated lyrics. A separator change does not silently overwrite them.

Tests cover content/config invalidation, cross-caller reuse, locking, corruption, cancellation, budget eviction, float export and short-clip overlap coverage. Listening preference on two songs is not a transcription-accuracy or alignment-accuracy benchmark.

## Validation results

The isolated host GPU smoke test used the production separator dependency versions and the full 212.27-second "22 Make" source. Initial preparation took 41.624 seconds. Two subsequent task reads across separate preprocessing sessions took 0.037 seconds total. The test disabled model construction after preparation and verified identical transcription and karaoke samples, with offline checkpoint loading. The vocal peak exceeded 1.0, confirming float export retained rather than clipped it. These timings exclude environment installation and are one local measurement, not a throughput guarantee.

The subsequent real-song melody test reused those cached vocals without loading a separator or importing Demucs. Preparing 44.1 kHz mono vocals and cached accompaniment took 0.569 seconds. MELODIA extraction took 2.840 seconds for vocals, 2.476 seconds for accompaniment and 2.665 seconds for the full mix. The maximum reconstruction difference between vocals plus accompaniment and the downmixed source was 1.19e-7. All three contours contained finite pitch values and usable pitch evidence. This verifies execution and signal consistency, not hum-search recall.

Analysis tests after the melody migration: 272 passed and 51 subtests passed. Another 28 database-dependent tests were skipped without a disposable database. Two unrelated failures were reproduced on unchanged HEAD and excluded from the final run: instrumental-confidence float equality in `test_curations.py`, and the pre-existing lead-in timestamp expectation in `test_karaoke_pipeline.py`. New cache/separator tests pass lint; both Dockerfiles pass hadolint.

GPU Docker execution on this host remains blocked by a stale NVIDIA runtime mount referencing `libnvidia-egl-wayland.so.1.1.21`, while the installed library is `.1.1.22`. Compute-only capabilities did not avoid it. No system GPU configuration was modified. The real GPU smoke test therefore ran in an isolated host virtual environment; container tests ran without GPU access. Repair that deployment issue before starting GPU containers. No live database or library reprocessing occurred during validation.

The Hugging Face model repository declares MIT in its [model card at the pinned revision](https://huggingface.co/KimberleyJSN/melbandroformer/blob/ac9b0614ab3cd7f77219e18ba494dfd93956c348/README.md). The author [confirms the MIT license and permission to use and redistribute the models](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/issues/18#issuecomment-4296256618), and separately [approved inclusion as an Audacity plugin](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/issues/7#issuecomment-2426784281). `vendor/roformer/NOTICE` preserves source provenance and these permission references. Weights are downloaded separately, not committed to Echora.

The default changed from overlap 8 to overlap 2 to reduce separation time. The measurements above describe overlap 8, not the new default. Separation-derived cache keys and karaoke/melody revisions change with overlap; existing transcript text remains protected.
