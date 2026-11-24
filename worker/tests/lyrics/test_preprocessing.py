from core.config import reset_config_cache

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest

import worker.lyrics.pipeline as lp
import worker.transcription.config as transcription_config
import worker.transcription.song as song_transcription
import worker.transcription.recovery as transcription_recovery


def setup(monkeypatch, provider, stored=None):
    events = []
    stored = stored or {}
    ids = {key: uuid.uuid4() for key in provider}
    cursor = Mock()
    last = {}

    def execute(sql, params=None):
        last.update(sql=sql, params=params)

    cursor.execute.side_effect = execute
    cursor.fetchall.return_value = [(ids[key], key, key) for key in provider]
    cursor.fetchone.side_effect = lambda: (
        stored.get(last["params"][0]) if "SELECT text," in last["sql"] else None
    )
    connection = Mock()
    connection.cursor.side_effect = lambda: nullcontext(cursor)
    monkeypatch.setenv("DATABASE_URL", "unused")
    reset_config_cache()
    monkeypatch.setattr(lp.psycopg, "connect", lambda _: nullcontext(connection))
    client = Mock()

    def lyrics(key):
        events.append(("retrieve", key))
        return provider[key]

    client.lyrics.side_effect = lyrics
    client.audio_bytes.side_effect = lambda key: key.encode()
    monkeypatch.setattr(lp, "NavidromeClient", lambda *args: nullcontext(client))
    monkeypatch.setattr(lp, "resolve_library_id", Mock(return_value=uuid.uuid4()))
    monkeypatch.setattr(
        lp, "plan_lyrics", Mock(return_value=SimpleNamespace(lyrics_external_ids=tuple(provider)))
    )
    for name in (
        "configure_representations",
        "_create_run",
        "start_attempt",
        "record_track",
        "finish_attempt",
        "release_model",
        "_store_embeddings",
    ):
        monkeypatch.setattr(lp, name, Mock())
    store = Mock(return_value=uuid.uuid4())
    monkeypatch.setattr(lp, "_store_lyrics", store)
    model = Mock()

    def load(*args):
        events.append(("embed-model", None))
        return model

    monkeypatch.setattr(lp, "LyricsEmbeddingModel", load)
    monkeypatch.setattr(transcription_config, "transcription_enabled", Mock(return_value=True))
    monkeypatch.setattr(
        transcription_config, "transcription_model", Mock(return_value=("model", "a" * 40))
    )

    def prepare(audio, **kwargs):
        events.append(("prepare", audio.decode()))

    monkeypatch.setattr(lp, "prepare_audio", Mock(side_effect=prepare))
    transcriber = Mock()
    # Generation phase: MOSS loads once (open), then generates each song's draft.
    transcriber.open.side_effect = lambda: events.append(("open-moss", None))
    transcriber.close.side_effect = lambda: events.append(("close-moss", None))

    def generate(audio, **kwargs):
        events.append(("transcribe", audio.decode()))
        return {"repairs": {}, "audio": audio}

    transcriber.generate.side_effect = generate
    transcriber.finish.side_effect = lambda draft: {
        "text": "recognized words",
        "status": "available",
        "ai_generated": True,
    }
    monkeypatch.setattr(lp, "SongTranscriber", Mock(return_value=transcriber))
    monkeypatch.setattr(lp, "diagnostic_writer", Mock())
    return SimpleNamespace(
        events=events,
        client=client,
        cursor=cursor,
        connection=connection,
        store=store,
        transcriber=transcriber,
        ids=ids,
        stored=stored,
    )


def test_resolve_all_lyrics_then_prepare_then_transcribe(monkeypatch):
    state = setup(
        monkeypatch,
        {
            "supplied": {"text": "known", "status": "available"},
            "missing-a": {"text": None, "status": "missing"},
            "missing-b": {"text": None, "status": "missing"},
            "instrumental": {"text": None, "status": "instrumental"},
        },
    )
    result = lp.backfill_lyrics("url", "user", "password")
    assert state.events == [("retrieve", name) for name in state.ids] + [
        ("prepare", "missing-a"),
        ("prepare", "missing-b"),
        ("open-moss", None),
        ("transcribe", "missing-a"),
        ("transcribe", "missing-b"),
        ("close-moss", None),
        ("embed-model", None),
    ]
    assert result["failed"] == 0
    assert result["embedded"] == 3
    assert result["instrumental"] == 1


def test_existing_text_and_disabled_transcription_do_not_prepare(monkeypatch):
    state = setup(
        monkeypatch,
        {
            "existing": {"text": None, "status": "missing"},
            "disabled": {"text": None, "status": "missing"},
        },
    )
    state.stored[state.ids["existing"]] = ("previous text", {}, "available")
    monkeypatch.setattr(transcription_config, "transcription_enabled", Mock(return_value=False))
    lp.backfill_lyrics("url", "user", "password")
    lp.prepare_audio.assert_not_called()
    state.client.audio_bytes.assert_not_called()
    state.transcriber.generate.assert_not_called()


def test_preparation_failure_is_counted_once_and_other_track_runs(monkeypatch):
    state = setup(
        monkeypatch,
        {
            "broken": {"text": None, "status": "missing"},
            "good": {"text": None, "status": "missing"},
        },
    )
    lp.prepare_audio.side_effect = [ValueError("bad audio"), None]
    result = lp.backfill_lyrics("url", "user", "password")
    assert result["failed"] == 1
    assert result["embedded"] == 1
    assert state.transcriber.generate.call_count == 1
    assert state.transcriber.generate.call_args.args == (b"good",)


def test_preparation_cancellation_propagates(monkeypatch):
    state = setup(monkeypatch, {"missing": {"text": None, "status": "missing"}})

    class Cancel(BaseException):
        pass

    lp.prepare_audio.side_effect = Cancel()
    with pytest.raises(Cancel):
        lp.backfill_lyrics("url", "user", "password")
    state.transcriber.generate.assert_not_called()


def test_disable_between_preparation_and_asr_stops_generation(monkeypatch):
    state = setup(monkeypatch, {"missing": {"text": None, "status": "missing"}})
    transcription_config.transcription_enabled.side_effect = [True, True, False]
    result = lp.backfill_lyrics("url", "user", "password")
    lp.prepare_audio.assert_called_once()
    state.transcriber.generate.assert_not_called()
    assert result["missing"] == 1


def test_concurrent_lyrics_writer_prevents_embedding_discarded_candidate(monkeypatch):
    state = setup(monkeypatch, {"missing": {"text": None, "status": "missing"}})
    state.store.return_value = None
    result = lp.backfill_lyrics("url", "user", "password")
    state.transcriber.generate.assert_called_once()
    lp._store_embeddings.assert_not_called()
    assert result["embedded"] == 0
