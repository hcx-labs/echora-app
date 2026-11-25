"""External AI requests to Claude through the official `anthropic` SDK.

Used when the External AI provider is "anthropic". Each caller keeps its own validation; this
module only returns the JSON text of one reply. Like the OpenAI-compatible path it runs with SDK
retries, redirects and environment proxies disabled and a finite deadline. The reply is
constrained to the caller's JSON schema with structured outputs, so the text always parses.
Errors carry a safe code only: never remote bodies, lyrics or keys.
"""

from __future__ import annotations

import anthropic
import httpx2

from core.errors import IntegrationError

DEFAULT_BASE_URL = "https://api.anthropic.com"


class ClaudeError(IntegrationError):
    """A failed Claude request, by code.

    `code` is one of: missing_key, endpoint_failure, transport_failure, refusal,
    incomplete_response, invalid_response.
    """

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def complete_json(
    settings,
    api_key: str | None,
    *,
    system: str,
    messages: list[dict],
    schema: dict,
    max_tokens: int,
    deadline_seconds: float,
    max_response_bytes: int,
    transport: httpx2.BaseTransport | None = None,
) -> str:
    """The JSON text of Claude's reply. `messages` use the Messages API content format."""
    if not api_key:
        # Without an explicit key the SDK would fall back to environment or profile credentials.
        raise ClaudeError("missing_key")
    try:
        with anthropic.Anthropic(
            api_key=api_key,
            base_url=settings.url or DEFAULT_BASE_URL,
            max_retries=0,
            http_client=httpx2.Client(follow_redirects=False, trust_env=False, transport=transport),
        ) as client:
            message = client.messages.create(
                model=settings.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                timeout=httpx2.Timeout(deadline_seconds, connect=5, pool=5),
            )
    except anthropic.APIStatusError:
        raise ClaudeError("endpoint_failure") from None
    except (anthropic.APIConnectionError, httpx2.HTTPError):
        raise ClaudeError("transport_failure") from None
    except anthropic.APIError:
        raise ClaudeError("invalid_response") from None
    if message.stop_reason == "refusal":
        raise ClaudeError("refusal")
    if message.stop_reason != "end_turn":
        raise ClaudeError("incomplete_response")
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text or len(text.encode()) > max_response_bytes:
        raise ClaudeError("invalid_response")
    return text
