"""Response models of the jobs category."""

from __future__ import annotations

from backend.api.responses import JobSnapshot, ResponseModel


class JobList(ResponseModel):
    """The user's root jobs, newest first."""

    jobs: list[JobSnapshot]


class JobBatch(JobSnapshot):
    """One batch (child job) of a sync."""

    batch_number: int
    track_count: int


class BatchPage(ResponseModel):
    """A page of a job's batches: running first, then queued, then the rest."""

    batches: list[JobBatch]
    total: int
