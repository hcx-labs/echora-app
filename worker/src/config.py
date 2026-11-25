"""Worker process configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting

# Every module that defines a settings class a worker reads, in catalog order. A worker loads
# them all when it starts (``core.config.load_all``).
CONFIG_MODULES = (
    "core.db.config",
    "core.jobs.config",
    "core.integrations.config",
    "core.library.config",
    "core.embeddings.config",
    "core.lyrics.config",
    "core.media.config",
    "worker.config",
    "worker.library.config",
    "worker.media.config",
    "worker.models.config",
    "worker.melody.config",
    "worker.fusion.config",
    "worker.remote.config",
    "worker.artwork.config",
    "worker.curation.config",
    "worker.transcription.config",
    "worker.kara.config",
    "worker.recordings.config",
)


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
