# Motion artwork (`worker.artwork`)

Looping videos rendered from album covers with LTX-2.5 on ComfyUI. The cover is pinned as the
first and last frame, so each clip loops seamlessly. It runs as the `motion_artwork` stage
(`worker/src/stages.py`), last in `navidrome_sync` and `import`, so the earlier stages have
released their models before ComfyUI needs the GPU.

## Flow

The stage's **prepare** hook, `prepare_prompts`, runs before the batch opens remote computation:
External AI writes the prompts first, so no GPU waits for them. They are keyed by external id,
cover SHA-256 and recipe hash.

`MotionArtworkProcessor` (`pipeline.py`), progress phase `motion-artwork`:

- **plan**: runs only when motion artwork is on and generated at the batch's location (during
  sync on this server, or on Modal). Otherwise `enabled` is false, and the batch summary has no
  `motion_artwork` key. It selects the batch's songs (`plan_tracks`) and the settings' recipe.
- **process**: fetches each cover and records it in `track_motion_artwork`. A cover that already
  has a complete loop is `reused`: any loop, or, with regenerate-outdated on, one of the current
  recipe. `mode="all"` replaces current loops too. The rest render in phases across every cover
  (`render_loops`); ComfyUI loads each large model once per phase and stops afterwards, so its
  memory returns to analysis. A prompt comes from `prepare_prompts`, from Gemma-4 in `auto`
  mode, or from the fixed prompt.
- **persist**: records each loop as complete or failed.

Summary `motion_artwork`: `tracks`, `rendered`, `reused`, `errors`, plus `timings`, and
`unavailable` when ComfyUI could not be reached.

## What it stores

- `motion_artworks`: one row per (cover SHA-256, recipe hash), with status, prompt, error, path,
  size, frames, fps, bytes and the recipe.
- `track_motion_artwork`: which cover a track has, and when it was checked.
- The video files themselves, under `ECHORA_MOTION_ARTWORK_DIR` (`MotionArtworkConfig` in
  `core.media.config`), apart from model storage.

A new recipe (any rendering setting) is a new hash. Older loops stay unless regenerate-outdated
is on.

## Configuration

- Generation settings (on/off, where to generate, recipe, prompt mode, regenerate-outdated) live
  in `motion_artwork_settings`, edited by administrators (`settings.py`).
- `ComfyUIConfig` (`config.py`), prefix `ECHORA_COMFYUI_`: the local ComfyUI install, its
  Python, the startup timeout, or the URL of an external ComfyUI.
- External prompts use the External AI settings (`writer.py`).

See [runtime settings](../../../docs/runtime-settings.md).

## Failures

- A cover that cannot be fetched fails with `InputUnavailableError`.
- A failed render is stored as failed (outcome `failed` with an output) and counted in `errors`.
- `ComfyUIUnavailable` is not an error: the stage stops and reports `unavailable`.
- `WriterError` (External AI prompt) and `ComfyUIError` are recorded per cover.
- A cover without a complete loop is planned again by the next sync.

## See also

- [Motion artwork](../../../docs/motion-artwork.md)
- [Modal compute](../../../docs/modal-compute.md)
