from echora_analysis.navidrome import NavidromeClient


def test_replace_playlist_updates_existing_id_without_creating_or_deleting() -> None:
    client = object.__new__(NavidromeClient)
    calls = []

    def request(method: str, **params: object):
        calls.append((method, params))
        return {"playlist": {"id": "existing"}}

    client._request = request

    result = client.replace_playlist("My list", ["one", "two"], "existing")

    assert result == "existing"
    assert calls == [("createPlaylist", {"songId": ["one", "two"], "playlistId": "existing"})]


def test_replace_playlist_creates_by_name_when_no_id_exists() -> None:
    client = object.__new__(NavidromeClient)
    calls = []

    def request(method: str, **params: object):
        calls.append((method, params))
        return {"playlist": {"id": "new"}}

    client._request = request

    assert client.replace_playlist("My list", ["one"], None) == "new"
    assert calls == [("createPlaylist", {"songId": ["one"], "name": "My list"})]


def test_catalog_exceeds_old_limit_and_probes_short_pages():
    client = object.__new__(NavidromeClient)

    def request(method, **params):
        offset = params["songOffset"]
        return {
            "searchResult3": {
                "song": [{"id": str(i)} for i in range(offset, min(offset + 211, 20003))]
            }
        }

    client._request = request
    assert len(client.all_tracks()) == 20003


def test_catalog_rejects_repetition_and_incomplete_responses():
    import pytest

    client = object.__new__(NavidromeClient)
    client._request = lambda *a, **k: {"searchResult3": {"song": [{"id": "same"}]}}
    with pytest.raises(RuntimeError, match="Repeating"):
        client.all_tracks()
    client._request = lambda *a, **k: {}
    with pytest.raises(RuntimeError, match="Incomplete"):
        client.all_tracks()


def test_batch_audio_cache_reuses_bytes_and_cleans_up():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from echora_analysis.navidrome import batch_audio_cache, _batch_state

    client = object.__new__(NavidromeClient)
    client.base_url, client.username = "https://music/", "user"
    client._auth = lambda: {}
    client.client = SimpleNamespace(
        get=Mock(
            return_value=SimpleNamespace(
                content=b"audio",
                headers={"content-type": "audio/flac"},
                raise_for_status=lambda: None,
            )
        )
    )

    class Cancelled(BaseException):
        pass

    import pytest

    with pytest.raises(Cancelled):
        with batch_audio_cache():
            directory = _batch_state.get()["directory"]
            assert client.audio_bytes("one") == client.audio_bytes("one") == b"audio"
            assert client.client.get.call_count == 1
            raise Cancelled()
    assert not directory.exists()
    assert _batch_state.get() is None


def test_audio_bytes_rejects_a_subsonic_error_instead_of_returning_it_as_audio():
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pytest

    client = object.__new__(NavidromeClient)
    client.base_url, client.username = "https://music/", "user"
    client._auth = lambda: {}
    error = {
        "subsonic-response": {
            "status": "failed",
            "error": {"code": 70, "message": "Song not found"},
        }
    }
    client.client = SimpleNamespace(
        get=Mock(
            return_value=SimpleNamespace(
                content=b"{}",
                headers={"content-type": "application/json"},
                json=lambda: error,
                raise_for_status=lambda: None,
            )
        )
    )
    with pytest.raises(RuntimeError, match="Song not found"):
        client.audio_bytes("dropped")


def test_lyrics_selects_main_and_preserves_all_structured_tracks():
    client = object.__new__(NavidromeClient)
    translation = {"kind": "translation", "lang": "es", "line": [{"value": "Hola"}]}
    main = {
        "kind": "main",
        "lang": "en",
        "synced": True,
        "line": [{"value": "Hello", "start": 1000}],
        "cueLine": [
            {
                "index": 0,
                "value": "Hello",
                "start": 1000,
                "end": 2000,
                "cue": [
                    {"value": "Hello", "start": 1000, "end": 2000, "byteStart": 0, "byteEnd": 4}
                ],
            }
        ],
    }
    calls = []

    def request(method, **params):
        calls.append((method, params))
        # Exercise Navidrome's opt-in: only enhanced requests include these layers.
        entries = (
            [translation, main]
            if params.get("enhanced")
            else [{"lang": "en", "line": main["line"]}]
        )
        return {"lyricsList": {"structuredLyrics": entries}}

    client._request = request
    result = client.lyrics("song")
    assert result["text"] == "Hello"
    assert result["language"] == "en"
    assert result["structured_lyrics"] == [translation, main]
    assert calls == [("getLyricsBySongId", {"id": "song", "enhanced": True})]


def test_translation_only_response_does_not_become_original_main_lyrics():
    client = object.__new__(NavidromeClient)
    translation = {"kind": "translation", "lang": "es", "line": [{"value": "Hola"}]}
    client._request = lambda *a, **k: {"lyricsList": {"structuredLyrics": [translation]}}
    result = client.lyrics("song")
    assert result["text"] is None
    assert result["status"] == "missing"
    assert result["structured_lyrics"] == [translation]


def _cue_line(index, value, cues, agent=None):
    data = value.encode()
    out, cursor = [], 0
    for word, start, end in cues:
        position = data.index(word.encode(), cursor)
        out.append(
            {
                "value": word,
                "start": start,
                "end": end,
                "byteStart": position,
                "byteEnd": position + len(word.encode()) - 1,
            }
        )
        cursor = position + len(word.encode())
    line = {"index": index, "value": value, "start": cues[0][1], "end": cues[-1][2], "cue": out}
    return {**line, "agentId": agent} if agent else line


def test_word_timed_lines_keep_spaces_backing_voice_and_offset():
    from echora_analysis.navidrome import word_timed_lines

    entry = {
        "synced": True,
        "offset": 100,
        "agents": [{"id": "v1", "role": "main"}, {"id": "v1-bg", "role": "bg"}],
        "line": [
            {"value": "Hello dear world (oh)", "start": 1000},
            {"value": ""},
            {"value": "散らかった", "start": 5000},
        ],
        "cueLine": [
            _cue_line(
                0,
                "Hello dear world",
                [("Hello", 1000, 1400), ("dear", 1400, 1800), ("world", 1800, 2400)],
                "v1",
            ),
            _cue_line(0, "(oh)", [("(oh)", 2000, 2600)], "v1-bg"),
            _cue_line(2, "散らかった", [("散", 5000, 5300), ("らかった", 5300, 6000)]),
        ],
    }
    lines = word_timed_lines(entry)
    assert [s["text"] for s in lines[0]["syllables"]] == ["Hello ", "dear ", "world ", "(oh)"]
    assert lines[0]["text"] == "Hello dear world (oh)"
    assert lines[0]["syllables"][-1]["voice"] == 2 and "voice" not in lines[0]["syllables"][0]
    assert lines[0]["start_ms"] == 900 and lines[0]["end_ms"] == 2500
    assert lines[1]["text"] == "散らかった" and lines[1]["syllables"][1]["start_ms"] == 5200


def test_word_timed_lines_need_every_line_timed():
    from echora_analysis.navidrome import word_timed_lines

    entry = {
        "synced": True,
        "line": [{"value": "One", "start": 0}, {"value": "Two", "start": 1000}],
        "cueLine": [_cue_line(0, "One", [("One", 0, 500)])],
    }
    assert word_timed_lines(entry) is None
    assert word_timed_lines({"line": [{"value": "One"}]}) is None
