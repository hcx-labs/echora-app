"""Errors of lyric translation."""

from core.errors import ProcessingError


class TranslationFailedError(ProcessingError):
    """The endpoint gave no usable translation. ``stage`` is the safe error code."""
