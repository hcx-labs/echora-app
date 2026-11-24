"""Worker process configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class WorkerConfig(EchoraConfig, env_prefix="ECHORA_WORKER_"):
    """Polling, heartbeats and batch size of a worker process."""

    poll_seconds: float = setting(2, gt=0, description="Wait between empty claims.")
    heartbeat_seconds: int = setting(10, gt=0, description="Under half the job lease.")
    schedule_check_seconds: int = setting(30, gt=0)
    cleanup_interval_seconds: int = setting(60, gt=0)
    shutdown_grace_seconds: int = setting(5, gt=0)
    batch_size: int = setting(
        128, gt=0, legacy="ECHORA_BATCH_SIZE", description="Songs per analysis batch."
    )
