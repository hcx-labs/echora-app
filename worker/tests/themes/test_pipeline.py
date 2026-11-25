import json
import uuid

import httpx
import httpx2
import numpy as np
import pytest

from worker.processing import ProcessingContext, run_stage

T = {n: uuid.UUID(f"00000000-0000-4000-8000-00000000000{n}") for n in (1, 2, 3)}

import worker.themes.pipeline as lyric_themes
import core.lyrics.translations as translation_storage
from core.integrations.external_ai import ExternalAISettings
from worker.themes.pipeline import (
    PROMPT_REVISION,
    ThemeError,
    ThemeProcessor,
    describe,
    lyrics_checksum,
    parse,
    search_text,
)

REPLY = {
    "description": "A narrator loves someone who never notices.",
    "themes": ["unrequited love", "longing"],
}


def claude(**values):
    return ExternalAISettings(
        **{
            "enabled": True,
            "provider": "anthropic",
            "model": "claude-haiku-5-5",
            "lyric_themes": True,
            **values,
        }
    )


def claude_reply(content, stop_reason="end_turn"):
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": json.dumps(content)}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def test_claude_request_and_search_text():
    seen = []

    def handle(request):
        seen.append(request)
        return claude_reply(REPLY)

    description, themes = describe(
        claude(), "k", "la la\n" * 5000, transport=httpx2.MockTransport(handle)
    )
    body = json.loads(seen[0].content)
    assert body["model"] == "claude-haiku-5-5"
    assert "untrusted lyrics data" in body["system"]
    assert body["output_config"]["format"]["schema"]["required"] == ["description", "themes"]
    assert len(body["messages"][0]["content"]) == lyric_themes.MAX_LYRICS_CHARS
    assert search_text(description, themes) == (
        "A narrator loves someone who never notices.\nThemes: unrequited love, longing"
    )


def test_claude_refusal_is_a_safe_code():
    with pytest.raises(ThemeError, match="^refusal$"):
        describe(
            claude(),
            "k",
            "x",
            transport=httpx2.MockTransport(lambda r: claude_reply({}, "refusal")),
        )


def test_openai_compatible_request():
    def handle(request):
        payload = json.loads(request.content)
        assert str(request.url) == "http://localhost:8000/v1/chat/completions"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][1]["content"] == "words"
        content = json.dumps(
            {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(REPLY)}}]}
        )
        return httpx.Response(200, content=content.encode())

    settings = ExternalAISettings(
        enabled=True, url="http://localhost:8000/v1", model="m", lyric_themes=True
    )
    assert (
        describe(settings, None, "words", transport=httpx.MockTransport(handle))[1]
        == REPLY["themes"]
    )
    with pytest.raises(ThemeError, match="^invalid_input$"):
        describe(settings, None, "  \n", transport=httpx.MockTransport(handle))


@pytest.mark.parametrize(
    "text", ["{", "null", '{"description":"","themes":[]}', '{"description":1,"themes":[]}']
)
def test_invalid_replies(text):
    with pytest.raises(ThemeError):
        parse(text)


def test_parse_cleans_tags():
    assert parse(
        json.dumps({"description": " a\n b ", "themes": ["x", " x ", "", 3] + ["t"] * 20})
    ) == (
        "a b",
        ["x", "t"],
    )


def test_checksum_is_hex_sha256_of_utf8():
    # Curation's SQL computes encode(sha256(convert_to(text, 'UTF8')), 'hex').
    assert (
        lyrics_checksum("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_overlay_needs_coverage_and_never_mixes_spaces(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused/unused")
    from core.curation.preview import THEME_COVERAGE_THRESHOLD, theme_overlay as _theme_overlay

    lyrics = np.ones((4, 2), np.float32)
    themes = np.full((4, 2), 2, np.float32)
    has_lyrics = np.array([True, True, True, False])
    few = np.array([True, False, False, False])
    matrix, available, active = _theme_overlay(lyrics, has_lyrics, themes, few)
    assert not active and matrix is lyrics and available is has_lyrics
    most = np.array([True, True, False, False])
    assert 2 >= THEME_COVERAGE_THRESHOLD * 3
    matrix, available, active = _theme_overlay(lyrics, has_lyrics, themes, most)
    assert active and matrix is themes
    # The lyric track without a description is neutral, not ranked by raw lyrics.
    assert available.tolist() == [True, True, False, False]
    assert not _theme_overlay(lyrics, np.zeros(4, bool), themes, most)[2]


class Embedded:
    def __init__(self, value):
        self.aggregate = np.full(1024, value, np.float32)


def describe_batch(ids, *, progress, describe_fn=None, embed=None):
    """The themes stage's summary for a batch of these Navidrome ids."""
    context = ProcessingContext(
        publish=progress, user_id="user", library=("http://n", "u", "p"), track_ids=tuple(ids)
    )
    processor = ThemeProcessor(describe_fn=describe_fn, embed=embed)
    return run_stage(processor, context).as_dict()


@pytest.fixture
def backfill(monkeypatch):
    state = {"settings": claude(), "saved": [], "rows": []}
    monkeypatch.setattr(translation_storage, "load_settings", lambda: (state["settings"], None))
    monkeypatch.setattr(lyric_themes, "_pending", lambda *a: state["rows"])
    monkeypatch.setattr(lyric_themes, "_save", lambda *a, **kw: state["saved"].append((a, kw)))
    return state


def test_backfill_disabled_does_nothing(backfill):
    backfill["settings"] = claude(lyric_themes=False)
    backfill["rows"] = [{"track_id": T[1], "text": "x", "title": "t"}]
    calls, progress = [], []
    result = describe_batch([], progress=progress.append, describe_fn=lambda *a: calls.append(a))
    assert result == {"described": 0, "failed": 0} and calls == [] and backfill["saved"] == []
    assert [update["message"] for update in progress] == ["Already current"]


def test_backfill_records_ready_and_failed(backfill):
    backfill["rows"] = [
        {"track_id": T[1], "text": "good", "title": "a"},
        {"track_id": T[2], "text": "refused", "title": "b"},
        {"track_id": T[3], "text": "unembeddable", "title": "c"},
    ]

    def fake_describe(settings, key, text):
        if text == "refused":
            raise ThemeError("refusal")
        return f"about {text}", ["tag"]

    def fake_embed(texts):
        assert texts == ["about good\nThemes: tag", "about unembeddable\nThemes: tag"]
        return [Embedded(0.5), RuntimeError("boom")]

    progress = []
    result = describe_batch(
        ["a"], progress=progress.append, describe_fn=fake_describe, embed=fake_embed
    )
    assert result == {"described": 1, "failed": 2}
    by_track = {args[0]: kw for args, kw in backfill["saved"]}
    assert by_track[T[1]]["description"] == "about good" and by_track[T[1]]["vector"][0] == 0.5
    assert by_track[T[2]] == {"code": "refusal"}
    assert by_track[T[3]] == {"code": "embedding_failed"}
    assert backfill["saved"][0][0][1] == lyrics_checksum("refused")
    assert progress[-1]["completed"] == 3 and progress[-1]["phase"] == "themes"


def test_prompt_revision_is_stable_text_hash():
    assert len(PROMPT_REVISION) == 16


def test_the_stored_theme_revision_follows_the_prompt():
    import hashlib

    from core.revisions import LYRIC_THEMES_REVISION
    from worker.themes.pipeline import SYSTEM

    assert hashlib.sha256(SYSTEM.encode()).hexdigest()[:16] == LYRIC_THEMES_REVISION
