import numpy as np
import pytest

import worker.voice.pipeline as pipeline
from worker.voice.pipeline import _aggregate_outputs


def test_voice_gender_keeps_vocal_presence_in_joint_scores() -> None:
    gender = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    voice = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)

    scores = _aggregate_outputs(gender, voice)

    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["instrumental"] == pytest.approx(0.45)
    assert scores["female"] == pytest.approx(0.425)
    assert scores["male"] == pytest.approx(0.125)


def test_instrumental_audio_does_not_get_decisive_gender_score() -> None:
    gender = np.array([[0.99, 0.01]], dtype=np.float32)
    voice = np.array([[0.98, 0.02]], dtype=np.float32)

    scores = _aggregate_outputs(gender, voice)

    assert scores["instrumental"] == pytest.approx(0.98)
    assert scores["female"] < 0.02
    assert scores["male"] < 0.001


def test_classifier_output_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="Unexpected classifier output shapes"):
        _aggregate_outputs(np.ones((2, 2)), np.ones((3, 2)))


def test_vocal_activity_keeps_patch_timestamps_and_reports_uncovered_tail():
    from types import SimpleNamespace
    from worker.voice.pipeline import VoiceGenderModel

    model = VoiceGenderModel.__new__(VoiceGenderModel)
    model._mel_patches = lambda _: np.zeros((2, 128, 96), dtype=np.float32)
    model._embedder = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input")],
        run=lambda *_: [np.zeros((2, 1280), dtype=np.float32)],
    )
    model._gender = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input")],
        run=lambda *_: [np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)],
    )
    model._voice = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input")],
        run=lambda *_: [np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)],
    )
    scores, activity = model.classify_with_activity(np.ones(5 * 16000, dtype=np.float32))
    assert sum(scores.values()) == pytest.approx(1)
    assert activity["windows"][0]["start_seconds"] == 0
    assert activity["windows"][1]["start_seconds"] == pytest.approx(2.048)
    assert activity["windows"][1]["end_seconds"] == pytest.approx(4.112)
    assert activity["windows"][1]["vocal_activation"] == pytest.approx(0.9)
    assert activity["unanalyzed_tail_seconds"] == pytest.approx(0.888)
    assert activity["confidence_calibrated"] is False


class _Cancelled(BaseException):
    pass


@pytest.mark.parametrize("ids", [(), ("shared-id",)])
def test_voice_scope_and_cleanup_on_cancellation(monkeypatch, ids):
    import uuid
    from unittest.mock import MagicMock

    from worker.processing import ProcessingContext, run_stage

    cursor = MagicMock()
    cursor.fetchall.return_value = [(uuid.uuid4(), "shared-id", "Title")]
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda *a: connection)
    monkeypatch.setattr(pipeline, "resolve_library_id", lambda *a: "library")
    monkeypatch.setattr(pipeline, "configure_representations", MagicMock())
    monkeypatch.setattr(pipeline, "_create_run", lambda *a: "run")
    monkeypatch.setattr(pipeline, "NavidromeClient", MagicMock())
    model = MagicMock()
    monkeypatch.setattr(pipeline, "_model", (model,))
    monkeypatch.setattr(pipeline, "shared_voice_model", lambda: model)

    def cancelled(*a):
        raise _Cancelled()

    monkeypatch.setattr(pipeline, "_fetch_stream", cancelled)
    context = ProcessingContext(user_id="user", library=("https://music", "u", "p"), track_ids=ids)
    with pytest.raises(_Cancelled):
        run_stage(pipeline.VoiceProcessor(), context)
    sql, params = cursor.execute.call_args_list[0].args
    assert "ts.library_id=%s" in sql and "ts.external_id=ANY(%s)" in sql
    assert params[:2] == ("library", list(ids))
    # The shared model is released, and nothing was half-written.
    assert pipeline._model is None
    connection.rollback.assert_not_called()


def test_unknown_library_selects_nothing(monkeypatch):
    from unittest.mock import MagicMock

    from worker.processing import ProcessingContext, run_stage

    connection = MagicMock()
    connection.__enter__.return_value = connection
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda *a: connection)

    def unknown(*a):
        raise ValueError("unknown library")

    monkeypatch.setattr(pipeline, "resolve_library_id", unknown)
    configure = MagicMock()
    monkeypatch.setattr(pipeline, "configure_representations", configure)
    context = ProcessingContext(user_id="u", library=("https://unknown", "u", "p"))
    with pytest.raises(ValueError, match="unknown library"):
        run_stage(pipeline.VoiceProcessor(), context)
    configure.assert_not_called()
    connection.cursor.assert_not_called()
