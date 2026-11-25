"""``GET /system/capabilities``."""

from __future__ import annotations

from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel


class ModelCapability(ResponseModel):
    """Whether a model is available and what it embeds."""

    status: str
    audio: bool
    text: bool


class CapabilityList(ResponseModel):
    """The embedding models this instance offers."""

    models: dict[str, ModelCapability]


class Capabilities(Endpoint):
    """List the embedding models and what each one embeds."""

    method = "GET"
    path = "/capabilities"
    response = CapabilityList

    def handle(self) -> dict[str, object]:
        return {
            "models": {
                "muq_mulan": {"status": "available", "audio": True, "text": True},
                "mert": {"status": "available", "audio": True, "text": False},
                "heartclap": {"status": "blocked-upstream-release", "audio": False, "text": False},
            }
        }
