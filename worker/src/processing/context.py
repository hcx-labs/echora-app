"""What a processor receives about the batch it runs in."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Progress", "ProcessingContext"]


@dataclass(frozen=True)
class Progress:
    """One progress update, in the shape the job queue stores and the web app reads."""

    phase: str
    message: str
    completed: int
    total: int
    unit: str = "tracks"
    summary: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The update as stored in the job's progress column."""
        update: dict[str, Any] = {
            "phase": self.phase,
            "message": self.message,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
        }
        if self.summary is not None:
            update["summary"] = dict(self.summary)
        return update


def _never_cancelled() -> None:
    return None


def _discard(_: dict[str, Any]) -> None:
    return None


@dataclass
class ProcessingContext:
    """The batch a stage runs in: cancellation, progress, where computation runs, and resources.

    ``resources`` holds what the batch opened for its stages (the database connection, the
    Navidrome client and library, the prepared-audio cache). Each stage reads what it needs
    through the typed accessors of the package that defines the resource.
    """

    check_cancelled: Callable[[], None] = _never_cancelled
    publish: Callable[[dict[str, Any]], None] = _discard
    location: Literal["local", "modal"] = "local"
    remote: Any = None
    """The batch's Modal session when computation runs remotely, else ``None``."""

    user_id: str | None = None
    resources: dict[str, Any] = field(default_factory=dict)

    def check(self) -> None:
        """Raise ``JobCancelled`` when the job was cancelled or its claim was lost."""
        self.check_cancelled()

    def report(self, progress: Progress) -> None:
        """Publish a progress update."""
        self.publish(progress.as_dict())

    def resource(self, name: str) -> Any:
        """A resource the batch opened. ``KeyError`` names a resource the batch lacks."""
        try:
            return self.resources[name]
        except KeyError:
            raise KeyError(f"This batch has no {name!r} resource") from None
