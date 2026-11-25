# Planning (`worker.planning`)

Decides which songs need which analysis, from what is already stored. It has no stage of its own:
sync parents use it to choose the songs they queue, and stages use it to choose the songs they
process.

## Flow

- `plan_audio()` (`audio.py`) returns an `AudioProcessingPlan`, the Navidrome ids that need
  MuQ-MuLan and MERT embeddings, chromaprints, melody contours, sound descriptors, waveforms,
  visual features and recording fingerprints. Library ingest calls it with
  `resolved_track_ids`, so the plan follows the tracks resolved from the bytes just downloaded,
  even if another worker remaps a source meanwhile. `download_external_ids` is the union.
  `audio_prerequisites()` names the decoded formats a song's outstanding phases need, for audio
  preparation.
- `plan_lyrics()` and `plan_karaoke()` return a `ProcessingPlan`: songs without current lyric
  embeddings, and songs whose karaoke timing is missing or from another pipeline or model
  revision. Word timing supplied with the lyrics counts as current when its digest matches.
- `select_sync_tracks()` (`sync.py`) picks the songs a `navidrome_sync` queues. `missing` mode
  picks unknown sources only. `all` mode adds:
  - sources due for revalidation (`freshness.py`);
  - songs the audio plan selects;
  - songs missing lyrics, lyric embeddings, audio profiles or voice evidence;
  - songs without lyrics when transcription is on;
  - songs the karaoke plan selects;
  - songs motion artwork still needs (`worker.artwork.settings.sync_selection`).
  All known songs are visited when hash checks are off or translations are enabled.
- `sources_needing_refresh()` (`freshness.py`): sources whose catalog metadata changed, that
  were never byte-verified, or whose last verification is at least
  `ECHORA_LIBRARY_SOURCE_RECHECK_SECONDS` old. With the default of 0, every known source counts as
  due.

## What it stores

Nothing; it only reads. A result is current when it matches:

- the revision constants in `core/src/revisions.py` (`MELODY_CONTOUR_REVISION`,
  `KARAOKE_PIPELINE_REVISION`);
- the library modules' revisions (`WAVEFORM_REVISION`, `VISUAL_FEATURE_REVISION`,
  `DESCRIPTOR_REVISION`);
- the configured model revisions (`EmbeddingConfig`);
- the recording encoder's representation.

Melody and karaoke are planned only while their features (`hum`, `karaoke`) are enabled.

## Configuration

No config class of its own. It reads `EmbeddingConfig`, `LibraryConfig`, the feature switches and
the recording encoder configuration. See [docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

Planning queries raise; the stage or the sync parent fails and the job is retried.

## See also

- [docs/batch-phases.md](../../../docs/batch-phases.md)
- [docs/analysis-correctness.md](../../../docs/analysis-correctness.md)
