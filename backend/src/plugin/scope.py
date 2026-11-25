"""What a plugin key may see: the songs of its user in its Navidrome server's library."""

from __future__ import annotations

import uuid
from typing import Any


def library_namespace(principal: dict[str, Any]) -> uuid.UUID:
    """The library namespace of the key's Navidrome server."""
    return uuid.uuid5(uuid.NAMESPACE_URL, principal["url"].rstrip("/"))
