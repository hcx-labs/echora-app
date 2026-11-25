"""``GET /sync/semantic-fusion``."""

from __future__ import annotations

from backend.api.deps import current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import SemanticFusionCounts
from core.db.connection import connect
from core.embeddings import store


class SemanticFusionStatus(Endpoint):
    """Count fused vectors and the lyrics and audio vectors behind them (whole corpus)."""

    method = "GET"
    path = "/semantic-fusion"
    response = SemanticFusionCounts
    dependencies = (current_user,)

    def handle(self) -> dict:
        with connect() as connection:
            return store.fusion_counts(connection)
