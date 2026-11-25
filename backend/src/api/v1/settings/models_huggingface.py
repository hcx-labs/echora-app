"""``PUT /settings/models/huggingface``."""

from __future__ import annotations

import httpx
from fastapi import Depends, HTTPException
from pydantic import Field, SecretStr

from backend.api.deps import SessionUser
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from backend.api.v1.settings._deps import admin_required
from worker.models.huggingface_token import TokenRejected, save


class HuggingFaceToken(PutRequest):
    """A Hugging Face token; blank clears the stored one."""

    token: SecretStr = Field(max_length=1000)


class HuggingFaceState(ResponseModel):
    """Whether a token is stored, and its account (never the token)."""

    has_token: bool
    account: str | None


class HuggingFaceTokenUpdate(Endpoint):
    """Store the Hugging Face token for gated model downloads, after checking it."""

    method = "PUT"
    path = "/models/huggingface"
    request = HuggingFaceToken
    response = HuggingFaceState

    def handle(
        self, request: HuggingFaceToken, user: SessionUser = Depends(admin_required)
    ) -> dict[str, object]:
        # Never echo the token or Hugging Face's response.
        try:
            account = save(request.token.get_secret_value())
        except TokenRejected:
            raise HTTPException(
                status_code=422, detail="Hugging Face did not accept this token"
            ) from None
        except httpx.HTTPError:
            raise HTTPException(
                status_code=503,
                detail="Could not reach Hugging Face to check the token. Try again.",
            ) from None
        return {"has_token": account is not None, "account": account}
