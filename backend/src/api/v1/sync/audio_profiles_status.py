"""``GET /sync/audio-profiles``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.sync._models import AudioProfileCoverage
from core.db.connection import connect
from core.embeddings import store
from core.revisions import AUDIO_PROFILE_REVISION, SUPPORTED_PROFILE_MODELS


class AudioProfilesStatus(Endpoint):
    """How many of the user's songs have a current audio profile, per model."""

    method = "GET"
    path = "/audio-profiles"
    response = AudioProfileCoverage

    def handle(self, user: SessionUser = Depends(current_user)) -> dict:
        models: dict[str, dict[str, int]] = {}
        with connect() as connection:
            for model_name in SUPPORTED_PROFILE_MODELS:
                total, current = store.audio_profile_coverage(
                    connection, user["id"], model_name, AUDIO_PROFILE_REVISION
                )
                models[model_name] = {
                    "total": total,
                    "current": current,
                    "missing": total - current,
                }
        return {
            "revision": AUDIO_PROFILE_REVISION,
            "models": models,
            "total": sum(value["total"] for value in models.values()),
            "current": sum(value["current"] for value in models.values()),
            "missing": sum(value["missing"] for value in models.values()),
        }
