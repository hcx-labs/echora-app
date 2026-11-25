"""``POST /sync/semantic-fusion``."""

from __future__ import annotations

from fastapi import Depends

import core.jobs.queue as jobs
from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import JobSnapshot


class SemanticFusionRebuild(Endpoint):
    """Rebuild the fused lyrics and audio vectors (one rebuild at a time, for the whole corpus)."""

    method = "POST"
    path = "/semantic-fusion"
    response = JobSnapshot
    status_code = 202

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        return jobs.enqueue(
            "semantic_fusion_build", "analysis", user["id"], dedupe_key="semantic_fusion_build"
        )
