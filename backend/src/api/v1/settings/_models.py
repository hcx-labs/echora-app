"""Request and response models shared by several settings endpoints."""

from __future__ import annotations

from typing import Literal

from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel


class ModelSwitch(PutRequest):
    """A model switch for syncs on this server (``local``) or on Modal."""

    enabled: bool
    location: Literal["local", "modal"] = "local"


class SwitchState(ResponseModel):
    """The stored switch."""

    enabled: bool
    location: str


class SwitchWithBacklog(SwitchState):
    """The stored switch, and how many songs the next sync would process."""

    pending: int
