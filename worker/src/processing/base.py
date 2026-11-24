"""The processor contract: one stage of library processing for a batch of songs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

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


@dataclass(frozen=True)
class ItemResult[InputT: Item, OutputT]:
    """What processing one item produced.

    Exactly one of ``output`` and ``error`` describes the result, except for outcomes that store
    nothing (``output`` and ``error`` both ``None``, as when an existing result is reused).
    ``outcome`` names the summary counter this item increments ("aligned", "reused", ...).
    """

    item: InputT
    outcome: str
    output: OutputT | None = None
    error: ProcessingError | None = None

    @classmethod
    def failed(cls, item: InputT, error: ProcessingError) -> ItemResult[InputT, OutputT]:
        """A result that records a failed item."""
        return cls(item=item, outcome="failed", error=error)


class Processor[ConfigT: EchoraConfig, InputT: Item, OutputT](ABC):
    """One processing stage.

    The runner calls, in order:

    1. ``plan``: the items that need this stage (reads only).
    2. ``open``: loads models and the like, once, and only when something was planned.
    3. ``process``: yields one ``ItemResult`` per planned item, in any order.
    4. ``persist``: stores each output as soon as it is yielded.
    5. ``close``: always, after ``open``, also on failure and cancellation.

    A failure that belongs to one item is yielded as ``ItemResult.failed`` and the stage goes on.
    An ``EchoraError`` escaping ``process`` fails every item that has no result yet. Any other
    exception is a bug and fails the job.
    """

    name: ClassVar[str]
    """Name of the stage in logs and progress."""

    summary_key: ClassVar[str]
    """Key of this stage's summary in the batch summary (read by the UI and the retry policy)."""

    counters: ClassVar[tuple[str, ...]] = ()
    """Counters that always appear in the summary, even at zero. Others appear once counted."""

    total_key: ClassVar[str] = "total"
    failure_key: ClassVar[str] = "failed"
    unit: ClassVar[str] = "tracks"
    activity: ClassVar[str] = "Processing"
    """Progress message prefix: "<activity> <item label>"."""

    current_message: ClassVar[str] = "Already current"
    """Progress message when nothing needs this stage."""

    def __init__(self, config: ConfigT) -> None:
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

    def persist(self, context: ProcessingContext, item: InputT, output: OutputT) -> None:  # noqa: B027
        """Store one output. Called for each result that has one, as soon as it is yielded.

        Any exception fails only this item; roll back what it wrote before raising.
        """

    def close(self) -> None:  # noqa: B027
        """Release what ``open`` loaded."""
