"""Run one processor over a batch: plan, open, process, persist, close, summarize."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from core.errors import EchoraError, ProcessingError

from .base import Item, Processor
from .context import ProcessingContext, Progress

__all__ = ["StageSummary", "run_stage"]

logger = logging.getLogger(__name__)


@dataclass
class StageSummary:
    """Counts of one stage, rendered with the stage's own keys."""

    processor: type[Processor[Any, Any, Any]]
    total: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    extra: dict[str, Any] = field(default_factory=dict)
    """Stage-specific facts beyond counts (a run id, a reason the stage could not run)."""

    @property
    def failed(self) -> int:
        """Items that failed."""
        return self.counts["failed"]

    def count(self, outcome: str, amount: int = 1) -> None:
        """Add to one outcome's counter."""
        self.counts[outcome] += amount

    def as_dict(self) -> dict[str, Any]:
        """The summary as the batch stores it: total first, then counters, then failures."""
        processor = self.processor
        rendered: dict[str, Any] = {}
        if processor.total_key is not None:
            rendered[processor.total_key] = self.total
        for name in processor.counters:
            rendered[name] = self.counts[name]
        for name, value in self.counts.items():
            if name != "failed" and name not in rendered:
                rendered[name] = value
        rendered[processor.failure_key] = self.failed
        rendered.update(self.extra)
        return rendered


def _log_failure(processor: Processor[Any, Any, Any], item: Item, error: BaseException) -> None:
    # Messages are not logged: provider messages can carry signed URLs or lyric text.
    if isinstance(error, ProcessingError):
        logger.warning("%s failed for %s at %s", processor.name, item.key, error.stage or "-")
    else:
        logger.error(
            "%s failed for %s: %s", processor.name, item.key, type(error).__name__, exc_info=False
        )


def run_stage(processor: Processor[Any, Any, Any], context: ProcessingContext) -> StageSummary:
    """Run one stage over the context's batch and return its summary.

    Items that fail are counted, logged by key and skipped. ``JobCancelled`` and programming
    errors propagate after ``close()``. When an ``EchoraError`` stops ``process``, the items
    without a result count as failed; when ``process`` stops early on its own (the feature was
    switched off mid-batch), they are left uncounted.
    """
    summary = StageSummary(type(processor))
    context.check()
    items = list(processor.plan(context))
    summary.total = len(items)
    if not items:
        context.report(Progress("planning", processor.current_message, 0, 0, processor.unit))
        return summary

    pending = {item.key: item for item in items}
    processor.open(context, items)
    try:
        for index, result in enumerate(processor.process(context, items), start=1):
            context.check()
            item = result.item
            pending.pop(item.key, None)
            if result.error is not None:
                _log_failure(processor, item, result.error)
                summary.count("failed")
            elif result.output is not None:
                try:
                    processor.persist(context, item, result.output)
                except Exception as error:  # noqa: BLE001 - one song's storage failure
                    _log_failure(processor, item, error)
                    summary.count("failed")
                else:
                    summary.count(result.outcome)
            else:
                summary.count(result.outcome)
            context.report(
                Progress(
                    processor.name,
                    f"{processor.activity} {item.label}" if item.label else processor.activity,
                    index,
                    len(items),
                    processor.unit,
                    summary.as_dict(),
                )
            )
    except EchoraError as error:
        logger.error(
            "%s stopped: %s (%d items without a result)",
            processor.name,
            type(error).__name__,
            len(pending),
        )
        summary.count("failed", len(pending))
    finally:
        processor.close()
    return summary
