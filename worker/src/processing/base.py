"""The processor contract: one stage of library processing for a batch of songs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from core.config import EchoraConfig
from core.errors import ProcessingError

if TYPE_CHECKING:
    from .context import ProcessingContext

__all__ = ["Item", "ItemResult", "Processor"]


class Item(BaseModel):
    """One unit of work: usually a song. Frozen, and complete enough to compute remotely."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    key: str
    """Stable identity for logs and summaries (a track id). Never a title or a path."""

    label: str = ""
    """What progress messages show (a song title)."""

    def track(self) -> dict[str, Any] | None:
        """The song a "started" progress update names (``id``, ``title``, ``artist``)."""
        return None


@dataclass(frozen=True)
class ItemResult[InputT: Item, OutputT]:
    """What processing one item produced.

    Exactly one of ``output`` and ``error`` describes the result, except for outcomes that store
    nothing (``output`` and ``error`` both ``None``, as when an existing result is reused).
    ``outcome`` names the summary counter this item increments ("aligned", "reused", ...).
    ``None`` leaves the count to what ``persist`` returns, for an item whose counter only storing
    tells (lyrics a concurrent writer superseded count as nothing).
    """

    item: InputT
    outcome: str | None
    output: OutputT | None = None
    error: ProcessingError | None = None

    @classmethod
    def failed(cls, item: InputT, error: ProcessingError) -> ItemResult[InputT, OutputT]:
        """A result that records a failed item."""
        return cls(item=item, outcome="failed", error=error)


class Processor[ConfigT: EchoraConfig | None, InputT: Item, OutputT](ABC):
    """One processing stage.

    The runner calls, in order:

    1. ``plan``: the items that need this stage (reads only).
    2. ``open``: loads models and the like, once, and only when something was planned.
    3. ``process``: yields one ``ItemResult`` per planned item, in any order.
    4. ``persist``: stores each output as soon as it is yielded.
    5. ``close``: always once ``open`` has started, also when ``open`` itself failed and on
       cancellation; release only what was acquired.

    A failure that belongs to one item is yielded as ``ItemResult.failed`` and the stage goes on.
    A ``ProcessingError`` escaping ``process`` fails every item that has no result yet. Any other
    exception (an integration that is down, a bug) fails the job, and the queue retries it.
    """

    name: ClassVar[str]
    """Name of the stage in logs and progress."""

    summary_key: ClassVar[str]
    """Key of this stage's summary in the batch summary (read by the UI and the retry policy)."""

    counters: ClassVar[tuple[str, ...]] = ()
    """Counters that always appear in the summary, even at zero. Others appear once counted."""

    total_key: ClassVar[str | None] = "total"
    """Key of the planned item count (``None``: the summary has no total)."""
    failure_key: ClassVar[str] = "failed"
    unit: ClassVar[str] = "tracks"
    activity: ClassVar[str] = "Processing"
    """Progress message: "<activity> <item label>", or the activity alone for unlabeled items."""

    current_message: ClassVar[str | None] = "Already current"
    """Progress message when nothing needs this stage (``None``: no update, for a step inside
    a larger stage)."""

    announces_items: ClassVar[bool] = False
    """Publish a progress update as each item starts, naming its song (``Item.track``), besides
    the one after it. For stages that take long per song and yield results in planned order."""

    storage_failure_fails_job: ClassVar[bool] = False
    """Whether an exception from ``persist`` fails the job (and so retries the batch) instead of
    counting one failed item. For stages whose output is costly and cannot be recomputed cheaply
    on the next sync, or whose failure counter the retry policy does not read."""

    def __init__(self, config: ConfigT = None) -> None:
        self.config = config

    @abstractmethod
    def plan(self, context: ProcessingContext) -> Sequence[InputT]:
        """Items of the batch that need this stage: missing or stale results."""

    def open(self, context: ProcessingContext, items: Sequence[InputT]) -> None:  # noqa: B027
        """Load models and do other one-off work for these items."""

    @abstractmethod
    def process(
        self, context: ProcessingContext, items: Sequence[InputT]
    ) -> Iterator[ItemResult[InputT, OutputT]]:
        """Compute the planned items and yield one result for each."""

    def persist(  # noqa: B027
        self, context: ProcessingContext, item: InputT, output: OutputT
    ) -> Iterable[str] | None:
        """Store one output. Called for each result that has one, as soon as it is yielded.

        Any exception fails only this item (the whole job with ``storage_failure_fails_job``);
        roll back what it wrote before raising. It may return
        counters to add on top of the item's outcome, for what only storing tells ("matched": the
        fingerprint joined a recording group).
        """

    def close(self) -> None:  # noqa: B027
        """Release what ``open`` loaded."""

    def facts(self) -> dict[str, Any]:
        """Stage-specific facts the summary carries beyond its counts (the run it wrote to)."""
        return {}
