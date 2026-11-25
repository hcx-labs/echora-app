"""Request bodies: unknown fields are rejected.

- ``PostRequest`` creates or performs something;
- ``PutRequest`` replaces a resource completely: every field is required unless it has a default;
- ``PatchRequest`` changes only the fields sent (``changes()``).

Strings are kept as sent: stripping a model wholesale would also strip passwords (``SecretStr``)
and lyrics. A field that should be trimmed says so with ``StrippedStr``.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints

StrippedStr = Annotated[str, StringConstraints(strip_whitespace=True)]


class RequestModel(BaseModel):
    """Base of every request body."""

    model_config = ConfigDict(extra="forbid")


class PostRequest(RequestModel):
    """A body that creates or performs something."""


class PutRequest(RequestModel):
    """A complete replacement of a resource."""


class PatchRequest(RequestModel):
    """A partial update: only the fields sent change."""

    def changes(self) -> dict[str, Any]:
        """The fields the client sent, with their values."""
        return self.model_dump(exclude_unset=True)
