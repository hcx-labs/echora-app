"""Errors of model downloads."""

from core.errors import ConfigurationError


class TokenRejected(ConfigurationError, ValueError):
    """Hugging Face refused the token."""
