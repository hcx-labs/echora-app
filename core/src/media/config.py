"""Where generated media lives."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class MotionArtworkConfig(EchoraConfig, env_prefix="ECHORA_MOTION_ARTWORK_"):
    """Motion artwork videos and their models. The worker renders them; the backend serves them."""

    dir: str = setting("/data/motion-artwork", description="Rendered videos (not model files).")
    models: bool = setting(
        False, description="Download the LTX-2.5 and Gemma files (about 44 GB) with the models."
    )
    render_timeout_seconds: float = setting(3600, gt=0)
    poll_seconds: float = setting(2, gt=0)


class KaraokeStemsConfig(EchoraConfig, env_prefix="ECHORA_KARAOKE_STEMS_"):
    """Karaoke stems: each song's vocals and instrumental, kept for the Stage player to remix.

    The worker writes them; the backend serves them. Unlike the preprocessing cache this is
    durable storage: put it on a persistent volume.
    """

    dir: str = setting("/data/karaoke-stems", description="Encoded stems (one file per song).")
    bitrate_kbps: int = setting(
        160,
        ge=96,
        le=256,
        description="Opus bitrate of each stereo pair; a stems file carries two (vocals and "
        "instrumental), so its total rate is twice this.",
    )
