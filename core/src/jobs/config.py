"""Job queue configuration."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from core.config import EchoraConfig, setting


class QueueConfig(EchoraConfig, env_prefix="ECHORA_JOB_"):
    """Leases, retries and attempts of durable jobs."""

    lease_seconds: int = setting(
        120, ge=15, legacy="ECHORA_WORKER_LEASE_SECONDS", description="How long a claim lasts."
    )
    retry_base_seconds: int = setting(
        2, gt=0, description="First retry delay; doubles per attempt."
    )
    retry_max_seconds: int = setting(60, gt=0, description="Longest retry delay.")
    max_attempts: int = setting(3, gt=0, description="Attempts before a job fails.")

    @model_validator(mode="after")
    def _retry_bounds(self) -> Self:
        if self.retry_base_seconds > self.retry_max_seconds:
            raise ValueError("Job retry base must be <= retry maximum")
        return self
