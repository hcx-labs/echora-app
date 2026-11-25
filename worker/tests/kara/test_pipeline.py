import io
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import worker.kara.pipeline as pipeline
from core.config import reset_config_cache
from worker.kara.aligner.request import AlignRequest
from worker.kara.pipeline import (
    FA_KARA_REVISION,
    _stored_model_revision,
    _validate_alignment_document,
    apply_adaptive_line_padding,
    build_lines_from_alignment_document,
)
from worker.processing import ProcessingContext, run_stage


def align_batch(progress=None, ids=("0", "1", "2")):
    """The karaoke stage's summary for a batch of these Navidrome ids."""
    context = ProcessingContext(
        publish=progress or (lambda update: None),
        user_id="user",
        library=("url", "user", "pass"),
        track_ids=tuple(ids),
    )
    return run_stage(pipeline.KaraokeProcessor(), context).as_dict()


REQUEST = AlignRequest(audio_path="i.audio", text="Hello", model_path="model")


def test_stored_model_revision_names_the_full_mix(monkeypatch):
    # The legacy separation switches no longer change what the aligner hears.
    for name in ("FA_KARA_VOCAL_SEPARATION", "FA_KARA_AUDIO_INPUT"):
        monkeypatch.setenv(name, "vocals")
    reset_config_cache()
    assert _stored_model_revision("model-revision") == f"{FA_KARA_REVISION}:model-revision:mix"


def test_alignment_document_restores_punctuation_and_spaces_to_display_syllables():
    document = {
        "alignment": {
            "lines": [
                {
                    "text": "Hello, world!",
                    "start_ms": 1000,
                    "end_ms": 1800,
                    "tokens": [
                        {"text": "Hello", "start_ms": 1000, "end_ms": 1300},
                        {"text": "world", "start_ms": 1400, "end_ms": 1800},
                    ],
                }
            ]
        }
    }

    assert build_lines_from_alignment_document(document) == [
        {
            "text": "Hello, world!",
            "start_ms": 1000,
            "end_ms": 1800,
            "syllables": [
                {"text": "Hello, ", "start_ms": 1000, "end_ms": 1300},
                {"text": "world!", "start_ms": 1400, "end_ms": 1800},
            ],
        }
    ]


def test_adaptive_padding_never_steals_time_from_previous_syllable():
    lines = [
        {
            "text": "uh-huh",
            "start_ms": 800,
            "end_ms": 2200,
            "syllables": [{"text": "uh-huh", "start_ms": 1000, "end_ms": 2000}],
        },
        {
            "text": "next",
            "start_ms": 1800,
            "end_ms": 3000,
            "syllables": [{"text": "next", "start_ms": 2000, "end_ms": 2400}],
        },
    ]

    result = apply_adaptive_line_padding(lines)

    assert result[0]["end_ms"] == 2000
    assert result[1]["start_ms"] == 2000
    assert result[0]["syllables"] == lines[0]["syllables"]


def test_adaptive_padding_uses_real_silence_and_syllable_duration():
    lines = [
        {"text": "first", "syllables": [{"text": "first", "start_ms": 1000, "end_ms": 1400}]},
        {"text": "second", "syllables": [{"text": "second", "start_ms": 2000, "end_ms": 2400}]},
    ]

    result = apply_adaptive_line_padding(lines)

    assert result[0]["end_ms"] == 1500
    assert result[1]["start_ms"] == 1900


def test_validate_alignment_document_accepts_monotonic_structured_output():
    document = {
        "schema_version": 1,
        "alignment": {
            "lines": [
                {
                    "source_index": 0,
                    "tokens": [{"start_ms": 100, "end_ms": 200, "ctc_score": 0.8}],
                }
            ]
        },
        "diagnostics": {"inference_passes": 1},
    }

    assert _validate_alignment_document(document) is document


def test_validate_alignment_document_rejects_reversed_tokens():
    document = {
        "schema_version": 1,
        "alignment": {
            "lines": [
                {
                    "source_index": 0,
                    "tokens": [
                        {"start_ms": 200, "end_ms": 300, "ctc_score": 0.8},
                        {"start_ms": 100, "end_ms": 150, "ctc_score": 0.9},
                    ],
                }
            ]
        },
    }

    try:
        _validate_alignment_document(document)
    except RuntimeError as error:
        assert "non-monotonic" in str(error)
    else:
        raise AssertionError("expected document validation failure")


@pytest.mark.parametrize("provided, source", [(False, "original_mix"), (True, "provided_audio")])
def test_worker_aligns_the_audio_as_given_without_line_times(
    monkeypatch, tmp_path, provided, source
):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    reset_config_cache()
    snapshot = (
        tmp_path
        / "hub"
        / f"models--{pipeline.DEFAULT_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / pipeline.DEFAULT_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    original = b"original compressed full mix"
    monkeypatch.setattr(pipeline, "_fa_kara_worker", lambda *args: object())

    def job(worker, request):
        assert Path(request.audio_path).read_bytes() == original
        # Synced lines give the line text; their start times never reach the aligner.
        assert request.text == "Hello"
        # The Echora Kara decoder defaults reach the aligner.
        assert request.exact_frame_clock and request.blank_free_tokens
        assert request.background_decoding
        assert (request.label_rules, request.fusion) == ("v3-sung", "on")
        # The mix recipe: onset-true timing, no loudness-based gap pull, inline backing placement.
        assert request.timing_offset_ms == 0
        assert not request.gap_onset_pull and request.inline_backing
        assert not request.optional_brackets
        assert (request.text_bonus, request.in_line_gap_cost) == (0.4, 0.2)
        return {
            "ok": True,
            "document": {
                "schema_version": 1,
                "alignment": {
                    "lines": [
                        {
                            "source_index": 0,
                            "text": "Hello",
                            "tokens": [
                                {
                                    "text": "Hello",
                                    "start_ms": 100,
                                    "end_ms": 200,
                                    "ctc_score": 0.9,
                                }
                            ],
                        }
                    ]
                },
            },
        }

    monkeypatch.setattr(pipeline, "_run_worker_job", job)
    result = pipeline._run_fa_kara(
        original, "Hello", "en", [{"text": "Hello", "start_ms": 100}], provided
    )
    assert result["diagnostics"]["audio_source"] == source
    assert result["diagnostics"]["separator"] is None


@pytest.mark.parametrize("cancel", [False, True])
def test_backfill_prewarms_before_alignment_and_skips_failed_tracks(monkeypatch, cancel):
    monkeypatch.setenv("DATABASE_URL", "unused-mocked")
    reset_config_cache()
    connection = MagicMock()
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda _: connection)
    connection.__enter__.return_value = connection
    tracks = [(uuid.UUID(int=i), str(i), str(i), "Hello", "en", []) for i in range(3)]
    connection.cursor.return_value.__enter__.return_value.fetchall.return_value = tracks
    client = MagicMock()
    client.__enter__.return_value = client
    events = []
    fetched = set()

    def audio(key):
        # The preparation pass fetches each song first; alignment rereads the source cache.
        data = key.encode()
        if key not in fetched:
            fetched.add(key)
            events.append(data)
        if key == "1":
            if cancel:
                raise KeyboardInterrupt()
            raise ValueError("bad audio")
        return data

    client.audio_bytes.side_effect = audio
    monkeypatch.setattr(pipeline, "NavidromeClient", lambda *args: client)
    monkeypatch.setattr(pipeline, "resolve_library_id", lambda *args: "library")
    monkeypatch.setattr(
        pipeline,
        "plan_karaoke",
        lambda *args, **kwargs: SimpleNamespace(karaoke_external_ids=["0", "1", "2"]),
    )
    monkeypatch.setattr(pipeline, "_stop_fa_kara_worker", lambda: events.append("stop"))

    def align(data, *args):
        assert events[:4] == ["stop", b"0", b"1", b"2"]
        assert data != b"1"
        events.append(("align", data))
        return {
            "lines": [],
            "diagnostics": {
                "audio_source": "original_mix",
                "separator_revision": None,
            },
            "model": "model",
            "model_revision": "revision",
            "alignment_document": {},
        }

    monkeypatch.setattr(pipeline, "_run_fa_kara", align)
    reports = []
    if cancel:
        with pytest.raises(KeyboardInterrupt):
            align_batch(reports.append)
        # The aligner process is stopped again when the stage ends.
        assert events == ["stop", b"0", b"1", "stop"]
    else:
        assert align_batch(reports.append) == {
            "total": 3,
            "aligned": 2,
            "failed": 1,
        }
        assert events[4:] == [("align", b"0"), ("align", b"2"), "stop"]
        assert [r["phase"] for r in reports][:7] == ["preprocess"] * 6 + ["models"]
        # Later phases reread the batch's source cache rather than copy raw audio
        # into a second unbounded per-karaoke directory.
        assert client.audio_bytes.call_count == 5


def test_backfill_aligns_the_original_mix_without_separating(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "unused-mocked")
    reset_config_cache()
    connection = MagicMock()
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda _: connection)
    connection.__enter__.return_value = connection
    tracks = [(uuid.UUID(int=i), str(i), str(i), "Hello", "en", []) for i in range(3)]
    connection.cursor.return_value.__enter__.return_value.fetchall.return_value = tracks
    client = MagicMock()
    client.__enter__.return_value = client

    def audio(key):
        if key == "1":
            raise ValueError("bad audio")
        return key.encode()

    client.audio_bytes.side_effect = audio
    monkeypatch.setattr(pipeline, "NavidromeClient", lambda *args: client)
    monkeypatch.setattr(pipeline, "resolve_library_id", lambda *args: "library")
    monkeypatch.setattr(
        pipeline,
        "plan_karaoke",
        lambda *args, **kwargs: SimpleNamespace(karaoke_external_ids=["0", "1", "2"]),
    )
    monkeypatch.setattr(pipeline, "_stop_fa_kara_worker", lambda: None)
    aligned = []

    def align(data, *args):
        aligned.append(data)
        return {
            "lines": [],
            "diagnostics": {"audio_source": "original_mix", "separator_revision": None},
            "model": "model",
            "model_revision": "revision",
            "alignment_document": {},
        }

    monkeypatch.setattr(pipeline, "_run_fa_kara", align)
    assert align_batch() == {
        "total": 3,
        "aligned": 2,
        "failed": 1,
    }
    assert aligned == [b"0", b"2"]


@pytest.mark.parametrize("mode", ["complete", "partial_cancel", "timeout"])
def test_worker_response_checks_cancellation_and_reaps(monkeypatch, mode):
    import os

    reader, writer = os.pipe()
    worker = MagicMock()
    worker.stdin = io.StringIO()
    worker.stdout = os.fdopen(reader, "r")
    worker.poll.return_value = None

    class Cancelled(BaseException):
        pass

    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if mode == "partial_cancel" and checks == 2:
            raise Cancelled()

    monkeypatch.setattr(pipeline, "get_check", lambda: check)
    try:
        if mode == "complete":
            os.write(writer, b'{"ok":true}\n')
            assert pipeline._run_worker_job(worker, REQUEST) == {"ok": True}
            assert checks >= 2
            worker.kill.assert_not_called()
        elif mode == "partial_cancel":
            os.write(writer, b'{"ok":')
            with pytest.raises(Cancelled):
                pipeline._run_worker_job(worker, REQUEST)
            worker.kill.assert_called_once()
            worker.wait.assert_called_once_with(timeout=5)
        else:
            with pytest.raises(TimeoutError):
                pipeline._run_worker_job(worker, REQUEST, timeout=0)
            worker.kill.assert_called_once()
            worker.wait.assert_called_once_with(timeout=5)
    finally:
        worker.stdout.close()
        os.close(writer)


def test_karaoke_lock_wait_checks_cancellation(monkeypatch):
    lock = MagicMock()
    lock.acquire.return_value = False
    monkeypatch.setattr(pipeline, "_KARAOKE_LOCK", lock)

    class Cancelled(BaseException):
        pass

    def check():
        raise Cancelled()

    processor = pipeline.KaraokeProcessor()
    with pytest.raises(Cancelled):
        processor.open(ProcessingContext(check_cancelled=check), [])
    lock.acquire.assert_called_once_with(timeout=0.25)
    processor.close()
    lock.release.assert_not_called()


def test_local_model_path_overrides_snapshot(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint-22000"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights v1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "missing-cache"))
    monkeypatch.setenv("FA_KARA_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("FA_KARA_REVISION", "v6-checkpoint-22000")
    reset_config_cache()
    monkeypatch.setattr(pipeline, "_fa_kara_worker", lambda *args: object())
    seen = {}

    def job(worker, request):
        seen["model"] = request.model_path
        return {
            "ok": True,
            "document": {
                "schema_version": 1,
                "alignment": {
                    "lines": [
                        {
                            "source_index": 0,
                            "text": "Hello",
                            "tokens": [
                                {
                                    "text": "Hello",
                                    "start_ms": 100,
                                    "end_ms": 200,
                                    "ctc_score": 0.9,
                                }
                            ],
                        }
                    ]
                },
            },
        }

    monkeypatch.setattr(pipeline, "_run_worker_job", job)
    try:
        result = pipeline._run_fa_kara(b"audio", "Hello", "en", [], False)
    finally:
        reset_config_cache()
    assert seen["model"] == str(checkpoint)
    # A local checkpoint is identified by its weights, not by the FA_KARA_REVISION label.
    assert result["model"] == str(checkpoint)
    assert result["model_revision"].startswith("local-")


def test_replacing_a_local_checkpoint_changes_its_identity(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights v1")
    monkeypatch.setenv("FA_KARA_REVISION", "same-label")
    monkeypatch.delenv("FA_KARA_MODEL_PATH", raising=False)
    reset_config_cache()
    try:
        # Without a local checkpoint, the pinned revision is the identity.
        assert pipeline.current_model_revision() == "same-label"
        monkeypatch.setenv("FA_KARA_MODEL_PATH", str(checkpoint))
        reset_config_cache()
        first = pipeline.current_model_revision()
        weights.write_bytes(b"weights v2, retrained")
        os.utime(weights, ns=(1, 1))
        second = pipeline.current_model_revision()
        # Same FA_KARA_REVISION label, new weights: planning sees a different aligner.
        assert first != second
        assert pipeline._stored_model_revision(first) != pipeline._stored_model_revision(second)
        (checkpoint / "model.safetensors").unlink()
        with pytest.raises(RuntimeError, match="no weights file"):
            pipeline.current_model_revision()
    finally:
        reset_config_cache()


def test_second_stream_words_that_overlap_are_marked_as_a_second_voice(monkeypatch):
    from worker.kara.pipeline import mark_backing_voice

    # Without inline backing placement an unbracketed second-stream phrase is a second voice only
    # when it is simultaneous; bracketed words always are.
    monkeypatch.setenv("FA_KARA_INLINE_BACKING", "false")
    reset_config_cache()
    lines = [
        {
            "text": "Lonely (so lonely) tonight",
            "syllables": [  # the second stream starts the echo before the lead's "lonely"
                {"text": "Lone", "start_ms": 1200, "end_ms": 1700},
                {"text": "ly (", "start_ms": 1700, "end_ms": 2500},
                {"text": "so ", "start_ms": 1000, "end_ms": 1500, "stream": 2},
                {"text": "lone", "start_ms": 1500, "end_ms": 1900, "stream": 2},
                {"text": "ly) ", "start_ms": 1900, "end_ms": 2400, "stream": 2},
                {"text": "tonight", "start_ms": 2600, "end_ms": 3200},
            ],
        },
        {
            "text": "Oh (yeah) yeah",
            "syllables": [  # bracketed on the lead path: backing all the same; stale mark cleared
                {"text": "Oh ", "start_ms": 4000, "end_ms": 4500},
                {"text": "(yeah) ", "start_ms": 4500, "end_ms": 5000},
                {"text": "yeah", "start_ms": 5000, "end_ms": 5500, "voice": 2},
            ],
        },
        {
            "text": "Call me now",
            "syllables": [  # unbracketed, on the second stream but after the lead: stays lead
                {"text": "Call ", "start_ms": 6000, "end_ms": 6500},
                {"text": "me ", "start_ms": 6520, "end_ms": 6900, "stream": 2},
                {"text": "now", "start_ms": 6900, "end_ms": 7300},
            ],
        },
        {
            "text": "Edges touch",
            "syllables": [  # an overlap the aligner did not place on its second stream
                {"text": "Edges ", "start_ms": 8000, "end_ms": 8600},
                {"text": "touch", "start_ms": 8300, "end_ms": 9000},
            ],
        },
    ]
    mark_backing_voice(lines)
    assert [s.get("voice") for s in lines[0]["syllables"]] == [None, None, 2, 2, 2, None]
    assert [s.get("voice") for s in lines[1]["syllables"]] == [None, 2, None]
    assert [s.get("voice") for s in lines[2]["syllables"]] == [None, None, None]
    assert [s.get("voice") for s in lines[3]["syllables"]] == [None, None]
    assert all("voice" not in line for line in lines)
    reset_config_cache()


def test_inline_backing_marks_every_second_stream_phrase_as_a_second_voice():
    from worker.kara.pipeline import mark_backing_voice

    reset_config_cache()  # inline backing placement is the default
    lines = [
        {
            "text": "I want something stronger (I want it)",
            "syllables": [  # the echo answers after the lead: no overlap, still the second voice
                {"text": "stronger (", "start_ms": 1000, "end_ms": 1400},
                {"text": "I ", "start_ms": 1420, "end_ms": 1500, "stream": 2},
                {"text": "want ", "start_ms": 1500, "end_ms": 1700, "stream": 2},
                {"text": "it)", "start_ms": 1700, "end_ms": 1900, "stream": 2},
            ],
        },
        {
            "text": "Oh (yeah) yeah",
            "syllables": [  # bracketed but left on the lead path by the aligner: still backing
                {"text": "Oh ", "start_ms": 4000, "end_ms": 4500},
                {"text": "(yeah) ", "start_ms": 4500, "end_ms": 5000},
                {"text": "yeah", "start_ms": 5000, "end_ms": 5500},
            ],
        },
    ]
    mark_backing_voice(lines)
    assert [s.get("voice") for s in lines[0]["syllables"]] == [None, 2, 2, 2]
    assert [s.get("voice") for s in lines[1]["syllables"]] == [None, 2, None]


def test_bracketed_words_and_lines_are_always_backing_vocals():
    from worker.kara.pipeline import mark_backing_voice

    lines = [
        {
            "text": "(One, two, three)",
            "syllables": [  # a whole bracketed line: every word, and the line itself
                {"text": "(One, ", "start_ms": 0, "end_ms": 400},
                {"text": "two, ", "start_ms": 400, "end_ms": 800},
                {"text": "three)", "start_ms": 800, "end_ms": 1200},
            ],
        },
        {
            "text": "Stay （ここに） with me",
            "syllables": [  # full-width brackets; the bracket spans syllables
                {"text": "Stay （", "start_ms": 2000, "end_ms": 2400},
                {"text": "こ", "start_ms": 2400, "end_ms": 2500},
                {"text": "こ", "start_ms": 2500, "end_ms": 2600},
                {"text": "に） ", "start_ms": 2600, "end_ms": 2800},
                {"text": "with ", "start_ms": 2800, "end_ms": 3000},
                {"text": "me", "start_ms": 3000, "end_ms": 3300},
            ],
        },
        {"text": "(Ooh)", "voice": None},  # no syllables: decided by the text
        {"text": "Plain line", "voice": "background"},  # stale line mark cleared
    ]
    mark_backing_voice(lines)
    assert [s.get("voice") for s in lines[0]["syllables"]] == [2, 2, 2]
    assert lines[0]["voice"] == "background"
    assert [s.get("voice") for s in lines[1]["syllables"]] == [None, 2, 2, 2, None, None]
    assert "voice" not in lines[1]
    assert lines[2]["voice"] == "background"
    assert "voice" not in lines[3]


def test_overlap_between_lines_is_not_a_second_voice():
    from worker.kara.pipeline import mark_backing_voice

    lines = [
        {"text": "First", "syllables": [{"text": "First ", "start_ms": 0, "end_ms": 2000}]},
        {
            "text": "Second",
            "syllables": [{"text": "Second ", "start_ms": 1500, "end_ms": 2500, "stream": 2}],
        },
    ]
    mark_backing_voice(lines)
    assert all("voice" not in s for line in lines for s in line["syllables"])


def test_worker_is_replaced_when_the_text_head_is_needed(monkeypatch, tmp_path):
    reset_config_cache()
    seen = []

    class Worker:
        def poll(self):
            return None

        def terminate(self):
            seen.append("terminated")

    def popen(command, **options):
        seen.append(options["env"])
        return Worker()

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    monkeypatch.setattr(pipeline, "_FA_KARA_WORKER", None)
    monkeypatch.setattr(pipeline, "_FA_KARA_WORKER_KEY", None)
    worker = pipeline._fa_kara_worker("revision")
    assert pipeline._fa_kara_worker("revision") is worker and len(seen) == 1
    # Another bonus still uses the text head: same worker. No bonus: a worker without it.
    monkeypatch.setenv("FA_KARA_TEXT_BONUS", "0.1")
    reset_config_cache()
    assert pipeline._fa_kara_worker("revision") is worker
    monkeypatch.setenv("FA_KARA_TEXT_BONUS", "0")
    reset_config_cache()
    assert pipeline._fa_kara_worker("revision") is not worker and "terminated" in seen
    reset_config_cache()


@pytest.mark.parametrize(
    "language, detects", [(None, True), ("", True), ("und", True), ("en", False), ("es", False)]
)
def test_lyrics_without_a_language_ask_the_aligner_to_identify_them(
    monkeypatch, tmp_path, language, detects
):
    """A known language the normalizer has no rules for ("es") stays on the generic path; only a
    missing one is identified by the aligner, which would otherwise spell English letter by letter.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    reset_config_cache()
    snapshot = (
        tmp_path
        / "hub"
        / f"models--{pipeline.DEFAULT_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / pipeline.DEFAULT_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "_fa_kara_worker", lambda *args: object())
    seen = []

    def job(worker, request):
        seen.append(request)
        line = {
            "source_index": 0,
            "text": "Hello",
            "tokens": [{"text": "Hello", "start_ms": 100, "end_ms": 200, "ctc_score": 0.9}],
        }
        return {"ok": True, "document": {"schema_version": 1, "alignment": {"lines": [line]}}}

    monkeypatch.setattr(pipeline, "_run_worker_job", job)
    pipeline._run_fa_kara(b"audio", "Hello", language, [], False)
    request = seen[0]
    assert bool(request.detect_language) is detects
    if detects:
        assert request.language == "auto"
        assert {"en", "ja", "ko"} <= set(request.detect_language)
    reset_config_cache()


class _Cancelled(BaseException):
    pass


def test_karaoke_scope_and_cleanup_on_cancellation(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(uuid.uuid4(), "shared-id", "Title", "words", "en", [])]
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda *a: connection)
    monkeypatch.setattr(pipeline, "resolve_library_id", lambda *a: "library")
    planner = MagicMock(return_value=SimpleNamespace(karaoke_external_ids=("shared-id",)))
    monkeypatch.setattr(pipeline, "plan_karaoke", planner)
    monkeypatch.setattr(pipeline, "_KARAOKE_LOCK", MagicMock())
    stops = MagicMock()
    monkeypatch.setattr(pipeline, "_stop_fa_kara_worker", stops)
    client = MagicMock()
    client.audio_bytes.side_effect = _Cancelled()
    monkeypatch.setattr(pipeline, "NavidromeClient", lambda *a: client)
    with pytest.raises(_Cancelled):
        align_batch(ids=("shared-id",))
    assert planner.call_args.args[2] == ["shared-id"]
    assert planner.call_args.kwargs == {"library_id": "library"}
    sql, params = cursor.execute.call_args.args
    assert "ts.library_id=%s" in sql and params == (["shared-id"], "library")
    assert stops.call_count == 2  # Before preparing, and when the stage ends.
    pipeline._KARAOKE_LOCK.release.assert_called_once()


def test_unknown_library_selects_nothing(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda *a: connection)

    def unknown(*a):
        raise ValueError("unknown library")

    monkeypatch.setattr(pipeline, "resolve_library_id", unknown)
    client = MagicMock()
    monkeypatch.setattr(pipeline, "NavidromeClient", lambda *a: client)
    with pytest.raises(ValueError, match="unknown library"):
        align_batch()
    client.audio_bytes.assert_not_called()
    connection.cursor.assert_not_called()


def test_supplied_word_timing_is_stored_without_aligning(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    track = uuid.uuid4()
    timing = [{"text": "Hello", "start_ms": 0}]
    connection.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        (track, "0", "Title", "Hello", "en", [], timing, "digest")
    ]
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda *a: connection)
    monkeypatch.setattr(pipeline, "resolve_library_id", lambda *a: "library")
    monkeypatch.setattr(
        pipeline, "plan_karaoke", lambda *a, **kw: SimpleNamespace(karaoke_external_ids=["0"])
    )
    monkeypatch.setattr(pipeline, "NavidromeClient", lambda *a: MagicMock())
    monkeypatch.setattr(pipeline, "_stop_fa_kara_worker", MagicMock())
    stored = []
    monkeypatch.setattr(
        pipeline, "_store_supplied_karaoke", lambda conn, *args: stored.append(args)
    )
    monkeypatch.setattr(pipeline, "_run_fa_kara", MagicMock(side_effect=AssertionError("aligned")))
    assert align_batch(ids=("0",)) == {"total": 1, "aligned": 0, "supplied": 1, "failed": 0}
    assert stored == [(track, timing, "digest")]
