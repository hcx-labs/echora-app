"""Errors of recording search and calibration."""

from core.errors import ConfigurationError, InputUnavailableError, SearchError


class CalibrationDisabled(ConfigurationError, ValueError):
    """Recording calibration is switched off."""


class TrackUnavailable(InputUnavailableError, ValueError):
    """A calibration song cannot be read."""


class QueueFull(SearchError, ValueError):
    """Too many recording searches are waiting."""
