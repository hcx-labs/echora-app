import json

import httpx2
import pytest
from pydantic import ValidationError

from core.integrations.claude import ClaudeError, complete_json
from core.integrations.external_ai import ExternalAISettings


def claude(**values):
    return ExternalAISettings(
        **{
            "enabled": True,
            "provider": "anthropic",
            "model": "claude-opus-5-5",
            "language_pairs": [{"source": "ja", "target": "en"}],
            **values,
        }
    )


def message(content, stop_reason="end_turn"):
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def replies(*responses, seen=None):
    queue = list(responses)

    def handler(request):
        if seen is not None:
            seen.append(request)
        return queue.pop(0)

    return httpx2.MockTransport(handler)


def test_claude_needs_no_url_but_openai_still_does():
    settings = claude()
    assert settings.url == "" and settings.ready()
    assert not claude(model="").ready()
    assert not claude(enabled=False).ready()
    with pytest.raises(ValidationError):
        ExternalAISettings(enabled=True, provider="openai", model="m")
    with pytest.raises(ValidationError):
        ExternalAISettings(provider="gemini")
    assert ExternalAISettings().provider == "openai"


def test_complete_json_rejects_oversized_reply():
    with pytest.raises(ClaudeError) as error:
        complete_json(
            claude(),
            "k",
            system="s",
            messages=[{"role": "user", "content": "u"}],
            schema={"type": "object"},
            max_tokens=10,
            deadline_seconds=5,
            max_response_bytes=5,
            transport=replies(message({"lines": "a long reply"})),
        )
    assert error.value.code == "invalid_response"
