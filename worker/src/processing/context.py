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
    track: Mapping[str, Any] | None = None
    """The song being worked on (``id``, ``title``, ``artist``), which the setup screen names."""

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
        if self.track is not None:
            update["track"] = dict(self.track)
        return update


def _never_cancelled() -> None:
    return None


def _discard(_: dict[str, Any]) -> None:
    return None


@dataclass
class ProcessingContext:
    """The batch a stage runs in.

    Its songs and library, cancellation, progress, where computation runs, and what earlier work
    prepared for it (``resources``).
    """

    check_cancelled: Callable[[], None] = _never_cancelled
    publish: Callable[[dict[str, Any]], None] = _discard
    location: Literal["local", "modal"] = "local"
    remote: Any = None
    """The batch's Modal session when computation runs remotely, else ``None``."""

    user_id: str | None = None
    library: tuple[str, str, str] | None = None
    """The batch's Navidrome connection: URL, user and password."""

    track_ids: tuple[str, ...] = ()
    """The batch's songs, as Navidrome ids."""

    options: Mapping[str, Any] = field(default_factory=dict)
    """Options of the job (``refresh_lyrics``, ``verify_audio_hashes``)."""

    resources: dict[str, Any] = field(default_factory=dict)

    def check(self) -> None:
        """Raise ``JobCancelled`` when the job was cancelled or its claim was lost."""
        self.check_cancelled()

    def report(self, progress: Progress) -> None:
        """Publish a progress update."""
        self.publish(progress.as_dict())

    @property
    def library_url(self) -> str:
        """The library's Navidrome URL; ``ValueError`` when the batch has no library."""
        if self.library is None:
            raise ValueError("This batch has no library connection")
        return self.library[0]

    def resource(self, name: str) -> Any:
        """A resource the batch opened. ``KeyError`` names a resource the batch lacks."""
        try:
            return self.resources[name]
        except KeyError:
            raise KeyError(f"This batch has no {name!r} resource") from None
