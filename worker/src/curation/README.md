# Curation (`worker.curation`)

Scheduled curation refreshes: rebuilds a user's curation from its recipe and publishes it as a
Navidrome playlist. It runs on the scheduled worker (`python -m worker.executor scheduled`), not
as an analysis stage. The selection itself (`core.curation.preview`) is shared with the backend's
preview.

## Flow

- `enqueue_due()` runs every `ECHORA_WORKER_SCHEDULE_CHECK_SECONDS` and queues one
  `curation_refresh` job per due curation (drafts, or enabled refreshes past `next_refresh_at`),
  at most `due_limit` per check. A curation's job is deduplicated by `curation:<id>`. The
  schedule advances only after the job is queued.
- `execute(job, context)` holds the curation's advisory lock (`locked()`, shared with the recipe
  routes) for the whole refresh, including the Navidrome calls:
  1. `selecting`: `_prepare()` selects the tracks and stores the plan as a `prepared`
     publication.
  2. `publishing`: replaces the playlist in Navidrome, then stores the playlist id immediately.
  3. Writes the revision and its tracks, then marks the curation `ready` and the publication
     `complete`.

  A job that finds a pending publication adopts it rather than selecting again, so a retried or
  replacement job publishes the same selection. A completed publication replays its summary.
- `enqueue()` queues a manual refresh; `assert_mutable()` guards recipe changes against a
  publication in progress.

The summary is `{curation_id, playlist_id, revision_number, track_count}`, or
`{curation_id, deleted: true}` when the curation is gone.

## What it stores

`curations` (status, schedule, last error, playlist id), `curation_publications` (the durable
publication intent and plan), `curation_revisions` and `curation_revision_tracks` (score and
evidence per track).

## Configuration

`CurationConfig` (`ECHORA_CURATION_`): `due_limit` (100), the curations queued per check. The
check interval is `WorkerConfig.schedule_check_seconds`. See
[docs/runtime-settings.md](../../../docs/runtime-settings.md).

## Failures

Any failure rolls back, marks the curation `failed` with the error and re-raises, so the job
fails retryably. A playlist creation whose outcome is unknown (`publishing` without a playlist
id) is never repeated automatically: the job raises until the remote playlist is reconciled.
`assert_mutable()` raises `ConflictError` and `enqueue()` raises `NotFoundError` for an unknown
curation (`core.errors`).

## See also

- [docs/jobs.md](../../../docs/jobs.md)
