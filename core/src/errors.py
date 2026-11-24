"""Exceptions shared by the backend and the worker.

Domains subclass these in their own ``errors`` module (``kara.errors.AlignmentError`` is a
``ProcessingError``). Messages are safe to log and to store on a job: they never carry provider
responses, URLs or credentials. Wrap a provider exception with ``raise ... from error`` instead of
formatting it into the message.

Cancellation is not an error: ``JobCancelled`` derives from ``BaseException`` so that the
``except Exception`` blocks that keep a batch going past one failed song never swallow it.
"""

from __future__ import annotations


class EchoraError(Exception):
    """Base class of every exception the analysis service raises on purpose."""


class ConfigurationError(EchoraError):
    """Configuration is missing or invalid (a model snapshot, a setting, a stored credential)."""


class IntegrationError(EchoraError):
    """An external system (Navidrome, External AI, Modal, ComfyUI) failed or refused a request."""


class RemoteComputeError(IntegrationError):
    """Model computation on Modal failed, or its deployment could not be prepared."""


class ProcessingError(EchoraError):
    """One item of one processing stage failed; the rest of the batch continues.

    ``stage`` names the processor and ``item`` the song (track or source id). Both are optional
    so that implementation code can raise without knowing where it runs; the runner fills them.
    """

    def __init__(self, message: str, *, stage: str | None = None, item: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.item = item


class InputUnavailableError(ProcessingError):
    """The item's input (audio, lyrics, cover) could not be read."""


class ModelUnavailableError(ProcessingError):
    """A model the stage needs is not installed or cannot be loaded."""


class NotFoundError(EchoraError):
    """The requested resource does not exist or is not visible to the caller."""


class ConflictError(EchoraError):
    """The request conflicts with the resource's current state (a refresh still running)."""


class SearchError(EchoraError):
    """A search request failed."""


class InvalidQueryError(SearchError):
    """A search request is malformed or out of range."""


class JobCancelled(BaseException):  # noqa: N818 - established name, not an error
    """Cooperative cancellation, or loss of the job's fenced claim."""
