import json

import httpx
import pytest

from echora_analysis.external_ai import DEFAULT_TRANSCRIPTION_PROMPT, ExternalAISettings
from echora_analysis.transcription_refinement import (
    CONTRACT,
    RefinementError,
    enabled,
    main_script,
    refine,
)

RESULT = {
    "text": "one\ntwo\nthree",
    "lines": [
        {"start_ms": 0, "end_ms": 1000, "text": "one", "speaker": "S01"},
        {"start_ms": 1000, "end_ms": 2000, "text": "two", "speaker": "S01"},
        {"start_ms": 2000, "end_ms": 3000, "text": "three", "speaker": "S01"},
    ],
    "ai_generated": True,
    "transcription": {"model": "moss"},
}


def settings(**values):
    return ExternalAISettings(
        **{"enabled": True, "url": "http://localhost:8000/v1", "model": "m", **values}
    )


def reply(content, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}]
            },
        )

    return httpx.MockTransport(handler)


def test_enabled_follows_external_ai_and_defaults_on():
    assert (
        ExternalAISettings().transcription_refinement
        and ExternalAISettings().transcription_prompt == DEFAULT_TRANSCRIPTION_PROMPT
    )
    assert enabled(settings())
    assert not enabled(settings(enabled=False))
    assert not enabled(settings(transcription_refinement=False))
    assert not enabled(settings(model=""))


def test_refine_keeps_timing_and_falls_back_per_line():
    seen = []
    out = refine(
        settings(transcription_prompt="Custom"),
        None,
        RESULT,
        title="T",
        artist="A",
        transport=reply(
            {
                "summary": "s",
                "lines": [
                    {"id": 0, "text": "won "},
                    {"id": 1, "text": ""},
                    {"id": 2, "text": "three"},
                ],
            },
            seen,
        ),
    )
    assert [l["text"] for l in out["lines"]] == ["won", "two", "three"]
    assert [l["start_ms"] for l in out["lines"]] == [0, 1000, 2000]
    assert out["text"] == "won\ntwo\nthree"
    record = out["transcription"]["refinement"]
    assert (
        record["changed_lines"] == 1 and record["kept_lines"] == 1 and record["song_summary"] == "s"
    )
    assert (
        record["transcribed_lines"] == RESULT["lines"] and out["transcription"]["model"] == "moss"
    )
    system, user = seen[0]["messages"]
    assert system["content"] == "Custom" + CONTRACT
    assert (
        json.loads(user["content"])["title"] == "T"
        and len(json.loads(user["content"])["lines"]) == 3
    )


@pytest.mark.parametrize(
    "content",
    [
        {"lines": [{"id": 0, "text": "a"}]},
        {"lines": [{"id": 2, "text": "a"}, {"id": 1, "text": "b"}, {"id": 0, "text": "c"}]},
        {"nope": 1},
    ],
)
def test_refine_rejects_wrong_shapes(content):
    with pytest.raises(RefinementError):
        refine(settings(), None, RESULT, title="T", artist="A", transport=reply(content))


def test_refine_endpoint_failure():
    transport = httpx.MockTransport(lambda request: httpx.Response(500, json={}))
    with pytest.raises(RefinementError):
        refine(settings(), "key", RESULT, title="T", artist="A", transport=transport)


def test_main_script():
    assert main_script("hello مرحبا مرحبا") == "ARABIC"
    assert main_script("") is None
