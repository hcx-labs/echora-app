"""A transcription resync becomes a forced transcription when a sync reaches the song."""

import uuid
from unittest.mock import MagicMock

import worker.lyrics.pipeline as pipeline
import worker.transcription.config as transcription
from core.resync import store as resync


def _processor(monkeypatch, enabled=True):
    track = uuid.UUID(int=1)
    monkeypatch.setattr(
        pipeline.store,
        "source_titles",
        lambda db, ids, library: [{"track_id": track, "external_id": "song", "title": "Song"}],
    )
    monkeypatch.setattr(resync, "requested", lambda db, name, ids: {track: {"language": "ja"}})
    done = MagicMock()
    monkeypatch.setattr(resync, "done", done)
    forced, discarded = MagicMock(), MagicMock()
    monkeypatch.setattr(pipeline.store, "force_transcription", forced)
    monkeypatch.setattr(pipeline.store, "discard_derived", discarded)
    monkeypatch.setattr(transcription, "transcription_model", lambda: ("moss", "rev"))
    monkeypatch.setattr(transcription, "transcription_enabled", lambda db: enabled)
    processor = pipeline.LyricsProcessor()
    processor.library_id = "library"
    return processor, track, forced, discarded, done


def test_a_transcription_request_forces_the_song_in_its_language(monkeypatch):
    processor, track, forced, discarded, done = _processor(monkeypatch)
    connection = MagicMock()
    assert processor._requested_transcriptions(connection, ["song"]) == {"song"}
    forced.assert_called_once_with(connection, track, "ja")
    discarded.assert_called_once_with(connection, track)
    done.assert_called_once_with(connection, track, "transcription")


def test_a_transcription_request_waits_for_ai_lyrics(monkeypatch):
    processor, _, forced, _, done = _processor(monkeypatch, enabled=False)
    assert processor._requested_transcriptions(MagicMock(), ["song"]) == frozenset()
    forced.assert_not_called()
    done.assert_not_called()
