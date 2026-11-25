"""``GET /sync/semantic-fusion``."""

from __future__ import annotations

from backend.api.deps import current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import SemanticFusionCounts
from core.db.connection import connect


class SemanticFusionStatus(Endpoint):
    """Count fused vectors and the lyrics and audio vectors behind them (whole corpus)."""

    method = "GET"
    path = "/semantic-fusion"
    response = SemanticFusionCounts
    dependencies = (current_user,)

    def handle(self) -> dict:
        with (
            connect(dict_rows=True) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT count(*) AS fused,
                          (SELECT count(*) FROM current_embeddings l WHERE l.embedding_type='lyrics'
                            AND l.window_index IS NULL) AS lyrics,
                          (SELECT count(*) FROM current_embeddings a WHERE a.embedding_type='audio-track'
                            AND a.window_index IS NULL) AS audio
                   FROM current_embeddings e WHERE e.embedding_type='semantic_fusion'"""
            )
            return cursor.fetchone()
