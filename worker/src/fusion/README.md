# Fusion (`worker.fusion`)

Semantic fusion: one vector per track from its lyrics (BGE-M3) and audio (MuQ-MuLan) track
vectors, for hybrid similarity. It runs as the `semantic_fusion_build` operation, stage
`semantic_fusion`. That operation is unbatched and reads no library.

## Flow

`build_semantic_fusion()` (`pipeline.py`):

1. Loads every track with both a current lyrics and a current audio vector (`loading`).
2. Whitens each modality with per-dimension statistics over that paired corpus. It then scales
   each side by the square root of its weight, concatenates them and normalizes the result.
3. Replaces the stored generation in one transaction (`writing`). It deletes the old fused
   vectors and superseded fusion runs, inserts the run and the vectors, and publishes the
   representation in `active_representation_specs`.

The stage reports `building` first. Its summary is `{total, fused}`, at the top level of the job.

Fusion stays a plain stage rather than a processor. It is one library-wide atomic rebuild: the
whitening statistics come from the whole corpus, so there is no per-song work to plan or retry.

## What it stores

`analysis_runs` (`kind='semantic_fusion'`, model `semantic_fusion`, revision `fusion-v1`),
`embeddings` (`embedding_type='semantic_fusion'`) and `active_representation_specs`. The
statistics and weights live in the run's config; its hash identifies the generation, so
`current_embeddings` never mixes two. Every build recomputes everything.

## Configuration

`FusionConfig` (`ECHORA_FUSION_`): `weight_lyrics` (0.75) and `weight_audio` (0.25). Their total
must be positive. See [docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

Any error rolls back the whole rebuild, so the previous generation stays current, and the job
fails retryably. With no paired tracks, the build stores nothing and returns `{total: 0, fused: 0}`.

## See also

- [docs/architecture.md](../../../docs/architecture.md)
