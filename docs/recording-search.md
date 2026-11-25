# Recording search

## Implementation status

The implementation adds microphone/upload capture, an independent temporal matcher, versioned fingerprint artifacts, same-worker query execution, and confirmed-match curation handoff. Recognition is disabled by default. The current local deployment enables recording search using a locally validated policy and disables new calibration collection.

The real NMFP triplet checkpoint has been exported and passed conversion parity. The operator approved its terms. The local bundle is `data/models/recording/nmfp-triplet-v1/`; weights remain outside git. `scripts/export_recording_model.py` reproduces the waveform-input ONNX artifact and evidence report from the pinned upstream checkpoint.

Recording search is enabled locally after replaying two labeled microphone recordings and one out-of-library clip against all 980 owner-visible references. Both positives were identified and the negative rejected using unchanged diagnostic thresholds. This small validation set does not establish a production false-positive rate. Conversion parity alone does not establish recognition accuracy.

## Scope

Recording search tries to identify the same recording in the user's indexed library. It does not search a worldwide catalog or promise to identify a concert performance against a studio recording. Hum search remains separate.

Only an identified top result enables "More like this" and "Start journey". Those actions load an authorized library track into the existing curation editor. The user still chooses and saves the recipe. Similarity and journeys continue using existing audio/lyrics representations, never fingerprint distance. An unidentified audio clip cannot become a curation seed in this release.

## Search from Browse

The waveform button beside Hum uses the same inline interaction: click to record a playing song, click again to stop and search, or let capture stop automatically after 20 seconds. A confirmed match appears as a playable track in the Browse list. Ambiguous clips show the ranked candidate tracks with a compact uncertainty notice. Candidates are labeled High-quality match, Possible match, or Low-quality match. There is no aggregate-score cutoff for displaying a candidate with sufficient temporal evidence. Only the strong, unambiguous top candidate can enable confirmed-match actions. Scores are raw similarity measures, not confidence percentages. A single ambiguous result may reflect an uncertain position within a song. No-match results show an explanation. There is no modal, calibration fallback, or separate index-status panel. Ordinary worker batching and queue order are unchanged.

The local policy and aggregate validation report are `data/models/recording/nmfp-triplet-v1/recording-quality-policy.json` and `recording-quality-validation.json`. They contain no raw clips. `ECHORA_RECORDING_MATCH_POLICY` points to the policy through the `/data` mount.

## One-shot diagnostic capture

An operator can arm `recording_diagnostic_captures` for a specific user only after that user explicitly requests preservation of their next recording. The consent version is `preserve-next-recording-v1`. The reservation expires after 24 hours if unused. The next admitted recording search atomically fills that user's reservation with a private raw-audio copy and its job ID, and sets a 24-hour expiry from capture. Subsequent uploads do not overwrite it or create further captures. No normal search response or job summary exposes this audio, and no public download endpoint exists.

Normal search audio still disappears on completion or cancellation. Worker maintenance deletes expired diagnostic copies at claim boundaries, so stopped workers or long claims can delay physical deletion. The operator can delete the diagnostic row earlier on request. This diagnostic copy has no ground-truth song label; do not treat the search prediction as the actual song. The inline frontend search flow and normal scheduling are unchanged.

## Earlier calibration collection

The temporary calibration panel has been removed from Browse. The authenticated calibration API and existing samples remain available under the same retention and deletion rules. The steps below describe the earlier capture flow.

1. Play a library song through speakers and record 10 to 15 seconds on your phone or computer. HTTPS or localhost is required for browser microphone access.
2. Stop and preview the clip locally. No upload occurs yet.
3. Select the exact library track, or choose **Not in my library** for a negative example. Optional notes can describe distance, speakers, or background noise.
4. Choose **Save calibration clip** to consent to storing the raw audio, including any speech it contains, for calibration.
5. Saved clips appear below the form and have a **Delete clip** action. Tell the assistant when a clip has been saved so it can evaluate it against the labeled track.

Start with a clear recording. Additional examples from farther away or with background noise help test rejection behavior. A handful of recordings can expose integration problems, but does not establish a reliable production false-match rate.

`ECHORA_RECORDING_CALIBRATION_ENABLED=true` enables this collection flow independently of recognition. The status API reports `recognition_enabled` and `calibration_enabled` separately. A calibrated recognition policy remains mandatory for `/search`. Once recognition is enabled, the dialog can still switch to calibration mode.

Saved raw files live at `/data/recording-calibration/<sample-id>.audio`, mapped to `data/recording-calibration/` on the host. The directory is mode 0700 and each file is mode 0600, owned by the analysis service UID. Neither filenames nor audio download URLs are exposed through the application. The database table `recording_calibration_samples` stores owner, user-provided label, notes, consent version, checksum, duration, and expiry. Source audio is not committed to git or sent to another service.

Collection requires the `X-Echora-Calibration-Consent: save-for-calibration-v1` header. Notes travel in a percent-encoded `X-Echora-Calibration-Notes` header instead of proxy URLs. API routes are authenticated and owner-scoped:

- `POST /library/recording/calibration/samples`, raw audio plus exactly one label: `expected_track_id` or `not_in_library=true`.
- `GET /library/recording/calibration/samples`, the owner's unexpired sample metadata.
- `DELETE /library/recording/calibration/samples/{sample_id}`, erase the owner's raw file and metadata.

Limits are 20 seconds and 8 MiB per clip, 20 retained clips per user, 200 globally, and six saves per user per minute. Known-track labels must be accessible to the owner at save time. Collection does not require a prebuilt fingerprint index. A private disk file is committed before its database row becomes visible; failed saves remove their file, and worker maintenance cleans up stale orphan files.

Samples expire after seven days. The existing analysis worker erases expired files at claim boundaries; stopped workers or long-running claims can delay physical deletion. List responses hide expired samples immediately. Explicit deletion unlinks the active file before reporting success. Backups, if configured, remain subject to deployment retention rules. Calibration examples are separate from ordinary search uploads, which still follow their shorter retention policy.

## Preprocessing comparison with AudioMuse

Reviewed [AudioMuse's Search by Recording documentation, section 17](https://neptunehub.github.io/AudioMuse-AI/ALGORITHM/#17-search-by-recording) against the pinned upstream frontend and Echora's exported graph. This comparison uses its documentation, not its matcher implementation.

- Both use the degradation-trained NMFP checkpoint, mono 8 kHz audio, one-second segments, and a half-second hop. Echora decodes directly to mono 8 kHz with FFmpeg for both library and query audio, rather than AudioMuse's native-rate loader followed by resampling.
- The exported graph includes the upstream Hann window and centered zero padding, 1024-point magnitude spectrum, 256-sample STFT hop, and 256 mel bands over 160 to 4000 Hz. Each segment produces 33 frames. Mel magnitudes have a 1e-5 floor, convert to decibels relative to the segment maximum, clip at -80 dB, and scale to [-1, 1]. Each resulting 128-dimensional fingerprint is L2-normalized. Export parity checks compare this frontend and the resulting embeddings with the pinned upstream implementation.
- AudioMuse documents RMS normalization before fingerprinting. Its general statement about level-sensitive frontends does not describe our segment-relative NMFP frontend above the magnitude floor. Raising the first microphone clip by about 35 dB left its alignment score unchanged to six decimal places. We therefore do not add RMS normalization, denoising, equalization, or source separation on this evidence. Extremely quiet signals near the magnitude floor still need testing; gain cannot repair a poor signal-to-noise ratio.
- Library fingerprints cover all complete one-second windows throughout the original mix, not a preview or the first two minutes. We do not remove silence or change playback speed, which would alter the timing used for alignment. Exact-silence windows contribute zero vectors. Clips shorter than one complete window cannot supply fingerprints.
- Echora deliberately limits query uploads to 20 seconds and 8 MiB, while AudioMuse allows larger uploads and a longer decoded excerpt. These are resource limits, not missing model preprocessing. Browser WebM/Opus goes through the same bounded FFmpeg query decoder.
- AudioMuse's product quantization, approximate index, and recognition thresholds are not preprocessing requirements. Echora retains float32 fingerprints and exact retrieval for calibration. We do not adopt its thresholds as evidence of accuracy on this library.

## Trigger full-library indexing from Sync

Open `/sync`, choose **Entire library**, then **Start processing**. Ordinary sync selects existing songs missing the current recording representation as well as new songs. **New tracks only** intentionally excludes existing tracks, so it cannot backfill their recording fingerprints. No separate index-building service or recognition policy is required.

The scan API reports recording-index counts for the selected connection and its current catalog. Sync does not display a separate recording-index panel. It separates usable fingerprints, unprocessed sources, and sources shorter than one fingerprint window. The worker reports a `recording_fingerprint` phase and a `recording_fingerprinted` result count. Existing current-version artifacts are reused, including fingerprints created during calibration checks. Failed artifacts remain missing and can be retried. Before executing missing audio work, ordinary sync rehashes the needed source downloads and refreshes their canonical identities, then replans analysis. Changed files therefore receive analysis for their current bytes rather than inheriting old artifacts. Access links move only for the exact changed source; other libraries and aliases remain intact. The recording writer still rejects a checksum mismatch if a source changes again after planning.

Sync still fills other missing representations, but recording-only work does not load MuQ or MERT. Sync uses the ordinary configured batch size regardless of whether the recording model is enabled. Completing the index does not enable recognition: calibration and full-library evaluation remain separate gates.

## Same-worker execution

Library synchronization adds `recording_fingerprint` to `AudioProcessingPlan` when a valid encoder manifest is configured. Existing source download and 8 kHz decoded-audio prerequisites are reused. Each complete fingerprint sequence commits independently. Its key is canonical track ID plus representation ID. Missing or changed representations are selected on the next Entire Library sync. Existing Chromaprint artifacts, recording groups, and SHA-256 track identity remain unchanged.

Queries are `recording_search` jobs on the existing analysis worker. The API does not load an inference runtime. A worker-owned claim subprocess runs ONNX on CPU, then retrieval and verification. The supervisor still owns lease renewal, process termination, and temporary-directory cleanup.

All analysis jobs use the existing queue order. Recording queries have no scheduling priority, and enabling the model does not change `ECHORA_BATCH_SIZE`, which defaults to 128 tracks. Already queued batches retain their original size; restarting sync is required to create batches with the corrected size. No response-time guarantee is made.

## Native model provisioning

Recording bundles are handled by the existing `python -m worker.models.download` command, alongside the other analysis models. Download and verification code lives in `worker.recordings.download`, not in deployment scripts. This support requires a release after 0.1.20.

Configure the provisioning process with:

```sh
ECHORA_RECORDING_MODEL_ID=hcX02/echora-nmfp-triplet-v1-onnx
ECHORA_RECORDING_MODEL_REVISION=956510e2ea26d22056d94c0ee500f1840fbd4719
ECHORA_RECORDING_MODEL_DIRECTORY=/data/models/recording/nmfp-triplet-v1
```

Then run the normal model-download command with network access. Source ID and revision are opt-in and must be configured together. Revisions must be full immutable commit hashes, not branches or tags. The directory must be an absolute path on the shared model volume. Without these source variables, existing manually installed bundles continue to work and no recording download occurs.

The downloader verifies a complete bundle in a temporary directory, then publishes it at `<directory>/<revision>`. It checks the model and parity checksums, license/parity attestations, representation ID and matcher policy. Existing valid bundles are reused without network access, including bundles installed by the earlier deployment bootstrap. Corrupt or incompatible caches fail closed and are not overwritten. Other generations remain untouched, and interrupted downloads remove their staging directory.

Point runtime `ECHORA_RECORDING_MODEL_MANIFEST` at `<directory>/<revision>/manifest.json` and `ECHORA_RECORDING_MATCH_POLICY` at `<directory>/<revision>/match-policy.json`. Kubernetes supplies these paths and the three download variables to the existing model provisioning setup; no script ConfigMap or extra download init container is needed. The normal Hugging Face token configuration also works for private bundles. The published bundle above is public and needs no token.

Serving and `--prune-only` never download or provision recording models. The model, manifest and policy must be available before starting offline workers. The published manifest includes the license acknowledgement for the approved deployment; other operators must review the upstream GPL terms and validation evidence before enabling it.

## Encoder contract

Set these variables identically on the API and analysis workers:

- `ECHORA_RECORDING_MODEL_MANIFEST`, path to a local manifest.
- `ECHORA_RECORDING_MATCH_POLICY`, path to a locally calibrated decision policy.

Compose forwards both variables and leaves them empty by default. Files under `/data/models/recording/` use the existing `/data` mount. Do not put model artifacts in git.

The encoder requires a self-contained ONNX graph with exactly one dynamic-batch float32 input `[batch, 8000]` and one output `[batch, 128]`. The graph includes audio preprocessing. A raw NMFP checkpoint or mel-input ONNX graph does not meet this contract. Our exporter imports the pinned upstream architecture and uses the official checkpoint, rather than using AudioMuse's exported model or implementation.

Example manifest structure, with placeholders that must be replaced and gates that must remain false until reviewed:

```json
{
  "schema_version": 1,
  "enabled": false,
  "license_acknowledged": false,
  "parity_validated": false,
  "model_path": "encoder.onnx",
  "model_sha256": "REPLACE_WITH_SHA256",
  "validation_report_path": "parity.json",
  "validation_report_sha256": "REPLACE_WITH_SHA256",
  "frontend": "embedded-waveform-v1",
  "sample_rate": 8000,
  "window_samples": 8000,
  "hop_samples": 4000,
  "embedding_dim": 128,
  "input_name": "waveform",
  "output_name": "embedding",
  "batch_size": 32
}
```

The adapter verifies checksums before loading model bytes and requires all three gates to be true. These gates are operator attestations, not automated proof of licensing or model parity. The report should identify source/checkpoint revisions, reference implementation, numerical error, retrieval parity, device/runtime versions, and test corpus. The graph must not rely on external tensor files.

The representation ID binds model bytes, input/output names, sample rate, window/hop geometry, and normalization semantics. Changing a model creates a new representation; it does not reinterpret old artifacts. Full one-second windows advance by half a second. Exact silence produces zero rows to preserve timing. Incomplete tails are dropped. A library track shorter than one second stores a terminal empty artifact and is not searchable.

## Independent temporal matcher

`recording_matcher.py` implements the matcher without AudioMuse source code:

1. Read only authorized reference sequences for the requested representation.
2. Compare distinct query windows to reference windows in bounded numeric blocks.
3. Vote for `reference_index - query_index`, with looser retrieval thresholds than final acceptance.
4. Verify shortlisted offsets and nearby offsets against consecutive windows of the complete query.
5. Require score, unique-window support, coverage, and temporal spread. Silence and missing overlap count against whole-query coverage.
6. Compare the best verified candidate against other recording identities, including near-miss candidates below acceptance thresholds. Known recording-group editions share an identity only for this margin check. They remain separate tracks.
7. Return `identified`, `ambiguous`, `no_match`, or `insufficient_audio`. Competing offsets within the winning track are conservatively ambiguous.

Scores are not probabilities. Thresholds are required, not supplied as production defaults.

The initial implementation performs exact segment comparisons, with a bounded offset shortlist. It does not yet include FAISS/HNSW, product quantization, or a separately published index generation. PostgreSQL stores float32 sequences as canonical, transactionally published artifacts. A server-side cursor reads references one track at a time. This avoids ANN filtering mistakes and provides a baseline against which future ANN candidate recall can be tested. It still scans the authorized corpus, so large libraries may exceed the 120-second retrieval budget.

Approximate raw storage for a four-minute track is 479 windows times 128 dimensions times 4 bytes, about 240 KiB before database overhead. Measure storage and latency before large-scale backfill.

## Reproducing the model export

Use a separate Python 3.11 environment. The export requirements retain Keras 2 compatibility for the upstream checkpoint; they are not runtime worker dependencies.

```bash
uv venv --python 3.11 /tmp/echora-nmfp-export
uv pip install --python /tmp/echora-nmfp-export/bin/python -r scripts/recording-export-requirements.txt
```

Check out `raraz15/neural-music-fp` at commit `15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc`. Download `nmfp-triplet.zip` from Zenodo record `15719945`. Its published MD5 is `ee8a3358fc5e5cdd09d6d2245d395021`. Keep the original archive alongside its extracted checkpoint directory. The exporter verifies the archive and checks each restored checkpoint file against its archive member before loading weights.

```bash
/tmp/echora-nmfp-export/bin/python scripts/export_recording_model.py \
  --source /path/to/neural-music-fp \
  --checkpoint /path/to/nmfp-triplet \
  --checkpoint-archive /path/to/nmfp-triplet.zip \
  --output data/models/recording/nmfp-triplet-v1 \
  --audio /path/to/local-music-a.flac \
  --audio /path/to/local-music-b.flac \
  --audio /path/to/local-music-c.flac \
  --license-acknowledged
```

Provide at least three distinct recordings, each with at least eight seconds of usable audio. They remain local. The script decodes at most the first 12 seconds of each, derives clean, quiet, noisy, and simulated-echo queries, and checks both embedding error and top recording/offset agreement against the TensorFlow reference. Silence, impulses, tones, and noise are additional numerical fixtures. This is export parity, not a phone-recording recognition benchmark.

The frontend derives fixed mel coefficients and the Hann window from the pinned Essentia reference. It implements the spectrum with real and imaginary DFT matrices, avoiding unsupported FFT/complex operations in deployment. It preserves the reference's 33 frames, magnitude mel calculation, amplitude floor, per-segment peak reference, and dynamic-range scaling.

An output directory is published only after every check passes. It contains `encoder.onnx`, `parity.json`, `manifest.json`, `UPSTREAM-LICENSE`, and the exact `exporter.py` used. The directory is readable by the non-root worker UID, and contains no source audio. The report records model/checkpoint/script checksums, source commit, runtime versions, numerical errors, and retrieval parity. Existing artifact directories are never overwritten. Recognition policy generation and service deployment are separate steps.

### Validated local artifact

- Archive SHA-256: `33e6059c0bf3ee3bc1df0479cefeb92dc85f4aa2a66833d430ce771aa0d7dceb`.
- Model SHA-256: `d48b86bbf6cb35d46716e1a491f2a7a42f4e51119f81c50b7c756eee3452b3e2`.
- Representation ID: `025c3107104551e3a68b6057672878c21d2964ad0f4d7f55be94adb796a6a3e3`.
- Model size: 75,850,225 bytes.
- Conversion check: 161 windows across 16 fixture cases, including three distinct local recordings.
- Maximum embedding absolute error against the float32 TensorFlow reference: `6.881915e-5`.
- Minimum embedding cosine agreement: `0.9999999688`.
- All 12 clean/quiet/noisy/simulated-echo queries preserved the top recording and offset. Both implementations selected the expected recording and offset in this small corpus.
- The actual adapter passed normalized-output, silence, dynamic-batch, and provenance tests. The existing analysis worker, UID 1001 and ONNX Runtime 1.23.2, successfully read and ran the graph through its current shared data mount.

These results compare the export with the reference in float32, not the upstream optional mixed-float16 execution mode. The fixture corpus is too small and lacks real microphone captures to calibrate production match thresholds.

Repeat the opt-in graph tests without changing production environment variables:

```bash
ECHORA_TEST_RECORDING_MANIFEST="$PWD/data/models/recording/nmfp-triplet-v1/manifest.json" \
python -m pytest worker/tests/recordings/test_encoder_real.py
```

The deployment now sets `ECHORA_RECORDING_MODEL_MANIFEST=/data/models/recording/nmfp-triplet-v1/manifest.json` on the existing mounted volume and enables calibration collection. The recognition-policy path now points to `recording-quality-policy.json`. No library-wide backfill was started during deployment.

## Decision policy

The policy is a JSON object with:

- `representation_id`, obtained from `recording_encoder.config_from_env().representation_id`.
- `matcher_revision`, currently `numpy-temporal-consensus-v3`. The deployed policy now uses v3.
- `calibrated: true`, only after held-out evaluation.
- `validation_dataset`, a nonempty identifier for that evaluation.
- `thresholds`, keyword arguments to `recording_matcher.MatchPolicy`.

Configuration fields are `window_similarity`, `min_score`, `min_support`, `min_coverage`, `min_temporal_spread`, and `min_margin`. Since matcher v2, the legacy `min_score` field is a high-quality boundary, not a candidate cutoff. Candidates still need independent supporting windows, coverage, and temporal spread. The top candidate is high quality only when it exceeds that band and has no competing identity or offset within the configured margin. Other candidates at or above the lower of `window_similarity` and `min_score` are possible matches; those below are low-quality matches. These are heuristic evidence bands, not calibrated confidence probabilities. Optional search parameters are `repeat_similarity`, `shortlist_size`, `offset_radius`, and `block_size`. See the dataclass for bounds. Tests use synthetic operating points, not deployable calibration values.

A queued query pins both representation ID and the SHA-256 of the policy file. If either changes before execution, the job fails rather than mixing configurations. There is one attempt per interactive query; users can retry with a fresh clip.

## API and privacy

- `GET /library/recording/status` reports availability and the user's indexed track count.
- `POST /library/recording/search` accepts raw audio and returns HTTP 202 with `job_id`.
- `GET /library/recording/search/{job_id}` returns owner-scoped status and completed results.
- `DELETE /library/recording/search/{job_id}` requests cancellation of an owned recording query.
- `GET /library/tracks?track_id=<uuid>` resolves a curation reference without scanning paginated library results. The existing visibility condition still applies.

Uploads have an 8 MiB limit and a 30-second receive deadline. Query decoding disables external file/network protocols and has a 30-second timeout. Queries are limited to 20 seconds, with 750 ms of capture/codec tolerance cropped off before inference. Microphone capture requests disabled noise suppression, echo cancellation, and automatic gain control, although browsers may not honor every constraint.

Admission allows one active query per user, six submissions per user per minute, and 32 active queries globally. These checks run in one serialized database transaction. Query bytes live in a private database table, not job payloads, logs, shared preprocessing caches, or public job summaries. Normal database access and backup policies still apply to these sensitive temporary bytes.

A database trigger erases audio when a job becomes terminal, including cancellation. Audio expires after 15 minutes and result rows after one day. The analysis supervisor performs cleanup at claim boundaries at most once per minute. A long-running claim or stopped worker delays expiry cleanup; terminal cleanup remains transactional. Configure separate database maintenance if strict wall-clock deletion is required.

Retrieval filters `user_track_links` before numeric comparisons. Result reads recheck visibility and do not promote a runner-up if the original winner has become inaccessible. Curation reloads the reference through the authorized library API. Cancellation and result publication use existing job-claim fencing.

## Validation before enabling

1. Record the approved code/checkpoint terms and artifact provenance.
2. Produce a waveform-input export and compare it with the reference on clean, quiet, noisy, codec-transcoded, and silent audio. Measure retrieval parity as well as numerical error.
3. Calibrate on held-out recordings from actual phones and speakers. Include absent songs, speech, silence, repeated sections, remasters, duplicate editions, and genuinely different live performances.
4. Measure false identification, rejection, correct identification, latency, memory, storage, and interactive wait while library processing runs.
5. Run migration, backfill, cancellation, restart, and access-revocation checks in staging.
6. Enable only after the false-identification operating point and worker capacity are acceptable.

Implementation checks passed on 169 selected backend tests, including PostgreSQL queue, authorization, worker, representation, and preprocessing tests. The full migration chain and revision 0044 downgrade/upgrade passed against a disposable database. Five frontend helper tests, TypeScript checking, and the production web build passed. ESLint reports one pre-existing hook-dependency warning in `PlayerProvider.tsx`.

These are targeted checks, not the complete application test suite. The encoder unit tests use a fake ONNX session, supplemented by opt-in tests against the actual exported model. Export provenance, encoder unit, and real-model tests pass together (42 tests). Calibration deployment added passing consent, label visibility, owner-only deletion, quota, disk failure, and expiry tests. The application and workers have been rebuilt and deployed through migration 0045 with calibration collection enabled. The production status check confirms collection enabled and recognition disabled. The first real microphone check has now completed. A 19.74-second labeled clip aligned around 11 seconds into its reference recording. All 38 query windows exceeded cosine 0.5 at that offset. Its mean alignment score was 0.686; the strongest of 19 tested distractors scored 0.160 under an exhaustive nonnegative-cosine alignment scan. The distractor set included 11 nearby tracks from the existing MuQ-MuLan space and eight additional tracks, including another song by the same artist. One additional planned reference was excluded because its downloaded bytes no longer matched its stored source checksum. Verified library fingerprints were retained under the model representation ID; query fingerprints were computed in memory only.

The clip's RMS level was about -53.4 dBFS. Raising its level to -18 dBFS changed the best alignment score by less than 0.000001, so this example does not justify adding gain normalization. The existing temporal matcher also returned the labeled track under an illustrative operating point. That operating point was not saved or enabled as a production policy. This is one positive example against a small selected reference set, not a full-library retrieval test or a false-positive-rate estimate. A follow-up evaluation used a second labeled song and an out-of-library clip against a combined set of 34 owner-visible reference tracks. The diagnostic thresholds remained unchanged. The second positive aligned at 5.5 seconds with score 0.648 and support from 36 of 38 query windows; its strongest competing track scored 0.170. The out-of-library clip returned `no_match`, with a maximum exhaustive alignment score of 0.154. The first positive still returned its correct track. These two positives and one negative establish a small integration check, not production calibration. A subsequent full-library test against 980 owner-visible references reproduced both positive identifications and the negative rejection. This local policy is now enabled for recording search. Recordings under more conditions are still needed to measure accuracy beyond this small set.

The matcher-v2 replay against 980 references retained both earlier positive identifications and the out-of-library rejection. The preserved Take a Hint clip now appears as a possible match at score 0.568 and offset 41.5 seconds, rather than disappearing below the old 0.600 acceptance cutoff. This four-clip check does not establish general noise robustness.


## Production review and deployment

The reviewed matcher is `numpy-temporal-consensus-v3`. It scores competing alignments independently of the window-support threshold. A close competitor below that threshold can therefore prevent a high-quality label. Repeated content counts once toward independent support, while every supported occurrence contributes to coverage and temporal spread.

The encoder caches verified artifact metadata for at most eight artifact signatures and one ONNX session per process. Cache keys include device, inode, size, modification time, and change time. Replacement triggers verification again. Validation streams file contents instead of retaining model bytes in the API cache. Silent input validates configuration without constructing a session. ONNX uses at most four intra-operation threads and one inter-operation thread. Cancellation interrupts decoder polling; the cooperative 120-second processing budget starts before preparation, not after encoding. Individual native initialization or inference calls still return before their next cooperative check; the worker supervisor remains responsible for process termination.

Migration `0047_source_membership` records authorized external source IDs separately from the deduplicated `user_track_links` projection. Full catalog snapshots replace only that user's memberships in that library. Batches add memberships without deleting sibling sources. Source remaps preserve surviving authorized aliases and rebuild affected owners under a per-library transaction lock. Migration backfill uses only proven existing links. A normal full catalog scan is required to recover aliases that earlier code already lost; global source rows alone cannot prove a user's access.

Migration `0048_source_freshness` records when downloaded bytes last verified a source identity. Entire library sync rehashes sources by default, including sources with complete artifacts. This adds download traffic but does not rerun valid analysis or change normal 128-track batching. New tracks only still excludes existing sources. Operators can set `ECHORA_SOURCE_RECHECK_SECONDS` above zero to defer unchanged-metadata sources between checks, at the cost of delayed detection of unreported byte changes. Metadata changes trigger a recheck regardless of that interval.

Each selected source resolves to a fixed canonical ID and checksum before analysis planning. Consumers validate downloaded bytes against that binding, even after a cache miss or concurrent source remap. An unavailable identity download no longer prevents healthy siblings from completing. Failed sources remain eligible for a later sync.

Recording results include their owner-visible playback connection. The frontend does not substitute the selected library for a recording result with no connection. Searches have explicit cancellation, stale-operation guards, bounded request retries, accessible feedback, and mobile-visible quality and position labels. Cancellation of a known server job is best-effort; a lost upload response can conceal the admitted job ID. Existing audio expiry and worker cleanup still apply.

The v3 replay against 980 references preserved two high-quality positive matches, rejected the out-of-library clip, and returned Take a Hint as a possible match. Each local replay took about 2 to 3 seconds. Four clips are not an accuracy or false-positive-rate estimate. No raw audio was copied or retention extended.

Deployment completed on 2026-09-22. The database is on `0048_source_freshness`; both workers and the web app were rebuilt and restarted. API and web health checks passed. The validated v3 policy replaced the contents of the existing `recording-quality-policy.json` path, so `.env` did not change. Batch size remains 128, strict source rechecking is enabled, and calibration collection remains disabled. No sync was started automatically.

Deployment procedure:

1. Back up the database, let active claims finish, and migrate through `0048_source_freshness`.
2. Set `ECHORA_RECORDING_MATCH_POLICY=/data/models/recording/nmfp-triplet-v1/recording-reviewed-policy.json`. The reviewed policy and `recording-reviewed-validation.json` are stored beside the model. Alternatively, install the reviewed policy at the already configured policy path, as this deployment did. Keeping the v2 policy with v3 code disables recognition by design. Fingerprints need no rebuild because the encoder representation is unchanged.
3. Deploy the backend and web together. Run an ordinary full sync to restore authorized source memberships from the current catalog.
4. Verify microphone permission cancellation, recording/search cancellation, navigation during search, mobile result layout, and playback across two connections in a real browser. Automated checks do not replace that smoke test.
