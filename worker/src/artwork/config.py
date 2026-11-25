"""ComfyUI configuration for motion artwork."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class ComfyUIConfig(EchoraConfig, env_prefix="ECHORA_COMFYUI_"):
    """The embedded ComfyUI, which the worker starts only while a batch renders loops."""

    dir: str = setting("/opt/comfyui")
    python: str = setting("/opt/comfyui-venv/bin/python")
    startup_timeout_seconds: float = setting(300, gt=0)
    in_worker_image: bool = setting(
        False,
        description="The worker image of this build carries ComfyUI (the GPU images set it), "
        "so the API can say so without having ComfyUI itself.",
    )
    url: str = setting(
        "", description="An already running ComfyUI instead of the embedded one (blank: embedded)."
    )
