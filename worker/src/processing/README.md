# Processing (`worker.processing`)

The framework every per-song stage is built on. A `Processor` describes one stage; `run_stage`
runs it over a batch and returns a `StageSummary`. Stages in `worker/src/stages.py` build
processors (`processor_stage`), and larger stages run several in sequence (library ingest,
lyrics).

## Flow

`run_stage(processor, context)` calls, in order:

1. `plan(context)`: the `Item`s of the batch that need the stage (missing or stale results). An
   empty plan skips the rest and reports `current_message` ("Already current" by default) in
   the `planning` phase; a processor inside a larger stage sets it to `None` to stay quiet.
2. `open(context, items)`: one-off work, such as loading a model, only when something was
   planned.
3. `process(context, items)`: yields one `ItemResult` per item, in any order.
4. `persist(context, item, output)`: stores each output as soon as it is yielded.
5. `close()`: always runs once `open` has started, including when `open` fails and on
   cancellation. It must release only what was acquired (`hasattr`/`getattr` guards).

An `ItemResult` has an `outcome` (the counter it increments: "aligned", "reused"), and an
`output` or an `error`:

- `ItemResult.failed(item, error)` counts as failed; the stage goes on.
- An output is persisted; the outcome is counted only if `persist` succeeds. If `persist` raises,
  the item counts as failed.
- `outcome=None` leaves the count to `persist`, which may return counters (an iterable of
  names). It may also return counters on top of an outcome ("matched" for a fingerprint that
  joined a recording group).
- A result with neither output nor error just counts its outcome (an existing result reused).
- An outcome of "failed" with an output still reaches `persist`, which can record the failure.

Every result publishes a `Progress` update in the processor's `name` phase: "<activity>
<label>", completed and total counts, unit, and the summary so far.

`ProcessingContext` carries the batch: user, library credentials, Navidrome track ids, job
options, the Modal session (`remote`) or `None`, cancellation (`check()`), progress (`publish`,
`report`) and `resources` from prepare hooks.

## Summaries

`StageSummary.as_dict()` renders, in order: `total_key` ("total", or nothing when `None`), the
declared `counters` (always present, even at zero), any other counted outcomes, `failure_key`
("failed"), then `facts()`. `facts()` adds stage-specific values such as the run a stage wrote
to; it is merged in even when the plan was empty. The UI and the executor's retry policy read
these keys, so they must stay stable.

## What it stores

Nothing itself; processors store their own results.

## Configuration

`Processor[ConfigT, InputT, OutputT]` takes an optional `EchoraConfig` as `config`.

## Failures

- `ProcessingError` (from `core.errors`) and its subclasses describe one item's failure. They are
  logged by item key and stage, never with their message, which may carry URLs or lyric text.
- An `EchoraError` escaping `process` stops the stage and counts every item still without a result
  as failed.
- `JobCancelled` and any other exception propagate (after `close`) and fail or stop the job.

## See also

- [docs/batch-phases.md](../../../docs/batch-phases.md)
- [docs/jobs.md](../../../docs/jobs.md)
