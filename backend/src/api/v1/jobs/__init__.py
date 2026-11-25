"""``/jobs``: the user's jobs, their batches, cancellation and dismissal."""

from backend.api.endpoint import Category
from backend.api.v1.jobs.batches import JobBatches
from backend.api.v1.jobs.cancel import CancelJob
from backend.api.v1.jobs.dismiss import DismissJob
from backend.api.v1.jobs.get import GetJob
from backend.api.v1.jobs.list import ListJobs

CATEGORY = Category(prefix="/jobs", endpoints=[ListJobs, GetJob, CancelJob, DismissJob, JobBatches])
