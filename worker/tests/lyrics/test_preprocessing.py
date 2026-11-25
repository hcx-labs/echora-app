import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import worker.lyrics.pipeline as lp
import worker.transcription.config as transcription_config
from core.config import reset_config_cache
from worker.processing import ProcessingContext


def run_lyrics():
    """The lyrics stage's summary for a batch of the planned songs."""
    from worker.processing import ProcessingContext

    return lp.lyrics_stage(ProcessingContext(user_id="user", library=("url", "user", "password")))


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
    result = run_lyrics()
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
    run_lyrics()
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
    result = run_lyrics()
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
        run_lyrics()
    state.transcriber.generate.assert_not_called()


def test_disable_between_preparation_and_asr_stops_generation(monkeypatch):
    state = setup(monkeypatch, {"missing": {"text": None, "status": "missing"}})
    transcription_config.transcription_enabled.side_effect = [True, True, False]
    result = run_lyrics()
    lp.prepare_audio.assert_called_once()
    state.transcriber.generate.assert_not_called()
    assert result["missing"] == 1


def test_concurrent_lyrics_writer_prevents_embedding_discarded_candidate(monkeypatch):
    state = setup(monkeypatch, {"missing": {"text": None, "status": "missing"}})
    state.store.return_value = None
    result = run_lyrics()
    state.transcriber.generate.assert_called_once()
    lp._store_embeddings.assert_not_called()
    assert result["embedded"] == 0


class _Cancel(BaseException):
    pass


def test_lyrics_scope_and_model_release_on_cancellation(monkeypatch):
    state = setup(monkeypatch, {"shared-id": {"text": "words", "status": "available"}})
    model = Mock()
    model.embed.side_effect = _Cancel()
    monkeypatch.setattr(lp, "LyricsEmbeddingModel", Mock(return_value=model))
    with pytest.raises(_Cancel):
        lp.lyrics_stage(
            ProcessingContext(
                user_id="user", library=("https://music", "u", "p"), track_ids=("shared-id",)
            )
        )
    assert lp.plan_lyrics.call_args.args[1] == ["shared-id"]
    library = lp.resolve_library_id.return_value
    assert lp.plan_lyrics.call_args.kwargs == {"library_id": library}
    # The track selection; per-track lyric lookups follow it.
    sql, params = next(
        call.args for call in state.cursor.execute.call_args_list if "track_sources" in call.args[0]
    )
    assert "ts.library_id=%s" in sql and params == (["shared-id"], library)
    lp.release_model.assert_called_once_with(model)
    lp.finish_attempt.assert_not_called()


def test_model_is_released_when_run_setup_fails(monkeypatch):
    setup(monkeypatch, {"shared-id": {"text": "words", "status": "available"}})
    model = Mock()
    monkeypatch.setattr(lp, "LyricsEmbeddingModel", Mock(return_value=model))
    lp._create_run.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        run_lyrics()
    lp.release_model.assert_called_once_with(model)


def test_unknown_library_selects_nothing(monkeypatch):
    state = setup(monkeypatch, {"shared-id": {"text": "words", "status": "available"}})
    lp.resolve_library_id.side_effect = ValueError("unknown library")
    with pytest.raises(ValueError, match="unknown library"):
        run_lyrics()
    state.client.lyrics.assert_not_called()
    lp.configure_representations.assert_not_called()
    state.cursor.execute.assert_not_called()


def test_new_transcripts_are_refined_before_embedding(monkeypatch):
    state = setup(monkeypatch, {"missing": {"text": None, "status": "missing"}})
    state.transcriber.finish.side_effect = lambda draft: {
        "text": "recognized words",
        "status": "available",
        "ai_generated": True,
        "lines": [{"text": "recognized words"}],
    }
    import core.lyrics.translations as translations
    import worker.transcription.refinement as refinement

    monkeypatch.setattr(translations, "load_settings", lambda: (SimpleNamespace(), None))
    monkeypatch.setattr(refinement, "enabled", lambda settings: True)
    monkeypatch.setattr(
        refinement,
        "refine",
        lambda settings, key, lyrics, **song: {**lyrics, "text": "corrected words"},
    )
    state.connection.execute.return_value = SimpleNamespace(
        fetchone=lambda: ("Artist",), rowcount=1
    )
    embedded = []
    model = Mock()
    model.embed.side_effect = lambda text: embedded.append(text)
    monkeypatch.setattr(lp, "LyricsEmbeddingModel", Mock(return_value=model))
    result = run_lyrics()
    assert (result["refined"], result["embedded"], result["failed"]) == (1, 1, 0)
    assert embedded == ["corrected words"]
