"""Recording extraction reuses the existing version-aware audio plan."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import worker.recordings.encoder as recording_encoder
from worker.planning.audio import plan_audio, audio_prerequisites


def test_recording_only_backfill_requests_shared_8khz_decode(monkeypatch):
    monkeypatch.setattr(
        recording_encoder, "config_from_env", lambda: SimpleNamespace(representation_id="new-model")
    )
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.side_effect = [
        [("song", "track", True, True, True, True, True, True, True)],
        [("song",)],
    ]
    plan = plan_audio(connection, "library", ["song"])
    assert plan.recording_fingerprint_external_ids == frozenset({"song"})
    assert plan.download_external_ids == frozenset({"song"})
    assert not plan.needs_muq and not plan.needs_mert and not plan.needs_melody
    assert audio_prerequisites(plan, "song").mono_rates == (8000,)
    assert audio_prerequisites(plan, "song").stereo_rates == ()
    assert cursor.execute.call_args.args[1] == (["song"], "library", "new-model")


def test_current_representation_and_disabled_encoder_need_no_work(monkeypatch):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    complete = [("song", "track", True, True, True, True, True, True, True)]
    monkeypatch.setattr(
        recording_encoder, "config_from_env", lambda: SimpleNamespace(representation_id="current")
    )
    cursor.fetchall.side_effect = [complete, []]
    assert not plan_audio(connection, "library", ["song"]).download_external_ids
    monkeypatch.setattr(recording_encoder, "config_from_env", lambda: None)
    cursor.fetchall.side_effect = [complete]
    assert not plan_audio(connection, "library", ["song"]).download_external_ids
