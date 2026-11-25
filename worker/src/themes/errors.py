"""Errors of lyric theme descriptions."""

from core.errors import IntegrationError


class ThemeError(IntegrationError):
    """The external AI gave no usable theme description."""
