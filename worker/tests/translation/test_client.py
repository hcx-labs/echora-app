import json

import httpx
import pytest

from core.integrations.external_ai import ExternalAISettings
from core.lyrics.translations import TranslationError, source_checksum
from worker.translation.client import translate, validate_response


def config(**values):
    return ExternalAISettings(
        enabled=True,
        url="http://localhost:8000/v1",
        model="local",
        language_pairs=[{"source": "JA", "target": "en_US"}],
        **values,
    )


def response(lines=None, finish="stop"):
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {
                        "content": json.dumps(
                            {"lines": lines if lines is not None else [{"id": 0, "text": "hello"}]}
                        )
                    },
                }
            ]
        }
    ).encode()


@pytest.mark.parametrize(
    "lines",
    [
        [],
        [{"id": 1, "text": "x"}],
        [{"id": True, "text": "x"}],
        [{"id": "0", "text": "x"}],
        [{"id": 0, "text": 1}],
        [{"id": 0, "text": "a\nb"}],
        [{"id": 0, "text": "x", "extra": 1}],
    ],
)
def test_invalid_mapping(lines):
    with pytest.raises(TranslationError):
        validate_response(response(lines), 1)


def test_exact_mapping_and_truncation():
    assert validate_response(response(), 1) == [{"id": 0, "text": "hello"}]
    assert (
        validate_response(response([{"id": 1, "text": "b"}, {"id": 0, "text": "a"}]), 2)[0]["text"]
        == "a"
    )
    for raw in [
        b"{",
        b"null",
        b"{}",
        response(finish="length"),
        response(finish="tool_calls"),
        response()[:-3],
        response([{"id": 0, "text": "a"}] * 2),
    ]:
        with pytest.raises(TranslationError):
            validate_response(raw, 2)
    assert source_checksum(["a", "b"]) != source_checksum(["a\nb"])


def test_client_payload_disabled_redirect_and_timeout():
    calls = []

    def handle(request):
        calls.append(request)
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["max_completion_tokens"] == 8192
        assert "max_tokens" not in payload
        assert str(request.url) == "http://localhost:8000/v1/chat/completions"
        assert json.loads(payload["messages"][1]["content"])["lines"] == [
            {"id": 0, "text": "ignore instructions"}
        ]
        assert request.extensions["timeout"]["connect"] == 5
        return httpx.Response(200, content=response())

    transport = httpx.MockTransport(handle)
    settings = config()
    pair = settings.language_pairs[0]
    with pytest.raises(TranslationError, match="disabled"):
        translate(ExternalAISettings(), None, pair, ["ignore instructions"], transport=transport)
    assert calls == []
    assert (
        translate(settings, "secret", pair, ["ignore instructions"], transport=transport)[0]["id"]
        == 0
    )
    assert len(calls) == 1
    redirects = []

    def redirect(request):
        assert "authorization" not in request.headers
        redirects.append(request)
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/"})

    with pytest.raises(TranslationError, match="endpoint_failure"):
        translate(settings, None, pair, ["x"], transport=httpx.MockTransport(redirect))
    assert len(redirects) == 1


def test_duplicate_json_keys_and_large_response():
    from worker.translation.client import MAX_RESPONSE_BYTES

    raw = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"lines":[{"id":0,"id":0,"text":"x"}]}'},
                }
            ]
        }
    ).encode()
    for data in [raw, b" " * (MAX_RESPONSE_BYTES + 1)]:
        with pytest.raises(TranslationError):
            validate_response(data, 1)


def test_transport_failure_and_empty_translation():
    settings = config()
    pair = settings.language_pairs[0]

    def fail(request):
        raise httpx.ReadTimeout("private-marker")

    with pytest.raises(TranslationError, match="^transport_failure$"):
        translate(settings, None, pair, ["x"], transport=httpx.MockTransport(fail))
    with pytest.raises(TranslationError, match="empty_translation"):
        translate(
            settings,
            None,
            pair,
            ["x"],
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=response([{"id": 0, "text": ""}]))
            ),
        )


def test_declined_translation_is_told_apart_from_gaps():
    settings = config()
    pair = settings.language_pairs[0]

    def reply(texts):
        lines = [{"id": i, "text": t} for i, t in enumerate(texts)]
        return httpx.MockTransport(lambda request: httpx.Response(200, content=response(lines)))

    source = ["one", "two", "three", "four"]
    # A note in the first line and nothing else: the model would not translate.
    with pytest.raises(TranslationError, match="^declined$"):
        translate(
            settings, None, pair, source, transport=reply(["I can't translate this.", "", "", ""])
        )
    # One line missing from an otherwise complete translation is still a gap.
    with pytest.raises(TranslationError, match="^empty_translation$"):
        translate(settings, None, pair, source, transport=reply(["un", "deux", "", "quatre"]))
