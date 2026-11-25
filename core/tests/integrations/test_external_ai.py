import json

import pytest
from pydantic import ValidationError

from core.integrations.external_ai import ExternalAISettings, LanguagePair


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
    "url",
    [
        "file:///tmp/x",
        "ftp://host",
        "https://user:pass@host",
        "http://host?",
        "http://host#",
        "http://host?q=x",
        "http://host/#x",
        "http://host:bad",
        "http://",
        "http://host\\x",
        "http://host/\n",
    ],
)
def test_invalid_url(url):
    with pytest.raises(ValidationError):
        ExternalAISettings(url=url)


def test_normalized_pairs_and_defaults():
    assert not ExternalAISettings().enabled
    assert config().language_pairs == [LanguagePair(source="ja", target="en-us")]
    for pairs in [[{"source": "en", "target": "EN"}], [{"source": "en", "target": "ja"}] * 2]:
        with pytest.raises(ValidationError):
            ExternalAISettings(language_pairs=pairs)
    assert (
        len(
            ExternalAISettings(
                language_pairs=[{"source": "en", "target": "ja"}, {"source": "fr", "target": "en"}]
            ).language_pairs
        )
        == 2
    )
    with pytest.raises(ValidationError):
        ExternalAISettings(enabled=True)
