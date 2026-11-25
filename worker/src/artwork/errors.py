"""Errors of motion artwork rendering."""

from core.errors import IntegrationError


class ComfyUIError(IntegrationError, RuntimeError):
    """ComfyUI rejected a graph, failed while running it, or is unreachable."""


class ComfyUIUnavailable(ComfyUIError):
    """ComfyUI is not installed here, could not start, or cannot be reached."""


class WriterError(IntegrationError, RuntimeError):
    """The endpoint failed or returned something that is not a usable prompt."""
