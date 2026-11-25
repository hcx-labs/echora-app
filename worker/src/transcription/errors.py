"""Errors of song transcription and its refinement."""

from core.errors import IntegrationError, ProcessingError


class WindowDecodeError(ProcessingError, ValueError):
    """A window's transcript could not be decoded; ``tokens`` counts what was generated."""

    def __init__(self, message, tokens=0):
        super().__init__(message)
        self.tokens = tokens


class RefinementError(IntegrationError):
    """The external AI gave no usable refined transcript."""
