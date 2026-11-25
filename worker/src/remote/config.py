"""Modal deployment configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class RemoteConfig(EchoraConfig, env_prefix="ECHORA_MODAL_"):
    """The image and GPU of the deployed Modal app."""

    image: str = setting(
        "ghcr.io/hackslashx/echora-worker:latest",
        description="Release images set the GPU image of their own version.",
    )
    gpu: str = setting("L40S", description="Set by the worker when it deploys.")
