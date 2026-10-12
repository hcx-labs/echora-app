from echora_analysis.settings import get_settings

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from echora_analysis import karaoke_pipeline as pipeline

from echora_analysis.karaoke_pipeline import (
    _anchored_source_lines,
    _stored_model_revision,
    apply_adaptive_line_padding,
    build_lines_from_alignment_document,
    _timed_source_lines,
    _validate_alignment_document,
    bound_to_synced_lines,
    guard_pathological_lead_ins,
    parse_ass_karaoke,
    stabilize_to_synced_lines,
)


def test_stored_model_revision_tracks_roformer_independent_of_legacy_env(monkeypatch):
    from echora_analysis.roformer import SEPARATION_REVISION

    for value in ("true", "false"):
        monkeypatch.setenv("FA_KARA_VOCAL_SEPARATION", value)
        get_settings.cache_clear()
        monkeypatch.setenv("FA_KARA_DEMUCS_MODEL", "ignored")
        get_settings.cache_clear()
        assert _stored_model_revision("model-revision").endswith(
            f":model-revision:roformer:{SEPARATION_REVISION}"
        )


def test_parse_ass_karaoke_preserves_line_and_syllable_timing():
    ass = (
        "[Events]\n"
        "Dialogue: 0,0:00:01.00,0:00:03.20,Default,,0,0,0,karaoke,{\\k20}{\\k50}Hel{\\k70}lo{\\k80}\n"
    )

    assert parse_ass_karaoke(ass) == [
        {
            "start_ms": 1000,
            "end_ms": 3200,
            "text": "Hello",
            "syllables": [
                {"start_ms": 1200, "end_ms": 1700, "text": "Hel"},
                {"start_ms": 1700, "end_ms": 2400, "text": "lo"},
            ],
        }
    ]


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


def test_lead_in_guard_matches_source_text_after_blank_lines():
    karaoke = [
        {
            "text": "'Cause everything",
            "start_ms": 72877,
            "end_ms": 77179,
            "syllables": [
                {"text": "'Cause ", "start_ms": 72877, "end_ms": 72977},
                {"text": "ev", "start_ms": 72977, "end_ms": 76719},
                {"text": "erything", "start_ms": 76719, "end_ms": 77179},
            ],
        }
    ]
    source = [
        {"text": "", "start_ms": 62660},
        {"text": "Other line", "start_ms": 68950},
        {"text": "'Cause everything", "start_ms": 76020},
    ]

    result = guard_pathological_lead_ins(karaoke, source, {})

    assert result[0]["start_ms"] == 76020
    assert result[0]["syllables"][0]["start_ms"] == 76020
    assert result[0]["syllables"][1]["start_ms"] == 76120
    assert result[0]["syllables"][2]["start_ms"] == 76719


def test_parse_ass_karaoke_ignores_headers():
    assert parse_ass_karaoke("[Script Info]\nTitle: example\n") == []


def test_timed_source_lines_filters_untimed_and_empty_entries():
    assert _timed_source_lines(
        [
            {"text": " First line ", "start_ms": 1000},
            {"text": "", "start_ms": 1500},
            {"text": "Untimed"},
            {"text": "Second line", "start_ms": 2200.9},
        ]
    ) == [
        {"text": "First line", "start_ms": 1000},
        {"text": "Second line", "start_ms": 2200},
    ]


def test_anchored_source_lines_interpolates_without_dropping_lyrics():
    assert _anchored_source_lines(
        [
            {"text": "One", "start_ms": 1000},
            {"text": "Missing"},
            {"text": "Three", "start_ms": 5000},
        ]
    ) == [
        {"text": "One", "start_ms": 1000, "interpolated": False},
        {"text": "Missing", "start_ms": 3000, "interpolated": True},
        {"text": "Three", "start_ms": 5000, "interpolated": False},
    ]


def test_stabilize_to_synced_lines_shifts_the_whole_line_not_only_first_syllable():
    karaoke = [
        {
            "text": "Learn to sign",
            "start_ms": 5500,
            "end_ms": 7000,
            "syllables": [
                {"text": "Learn", "start_ms": 5700, "end_ms": 6000},
                {"text": " to", "start_ms": 6000, "end_ms": 6300},
                {"text": " sign", "start_ms": 6300, "end_ms": 6800},
            ],
        }
    ]

    result = stabilize_to_synced_lines(karaoke, [{"text": "Learn to sign", "start_ms": 6500}])

    assert result[0]["start_ms"] == 6500
    assert result[0]["syllables"] == [
        {"text": "Learn", "start_ms": 6500, "end_ms": 6800},
        {"text": " to", "start_ms": 6800, "end_ms": 7100},
        {"text": " sign", "start_ms": 7100, "end_ms": 7600},
    ]


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


def test_guard_pathological_lead_ins_delays_only_stretched_first_syllable():
    diagnostics = {"source_offset_ms": 0, "source_drift_ms_per_minute": 0}
    result = guard_pathological_lead_ins(
        [
            {
                "text": "Siento frío",
                "start_ms": 197680,
                "end_ms": 205110,
                "syllables": [
                    {"text": "Sien", "start_ms": 197880, "end_ms": 203830},
                    {"text": "to", "start_ms": 203990, "end_ms": 204470},
                ],
            }
        ],
        [{"text": "Siento frío", "start_ms": 204010}],
        diagnostics,
    )

    assert result[0]["start_ms"] == 204010
    assert result[0]["syllables"] == [
        {"text": "Sien", "start_ms": 204010, "end_ms": 204110},
        {"text": "to", "start_ms": 204110, "end_ms": 204470},
    ]
    assert diagnostics["pathological_lead_in_guarded_lines"] == [0]


def test_bound_to_synced_lines_clamps_syllables_to_source_window():
    karaoke = [
        {
            "text": "First line",
            "start_ms": 800,
            "end_ms": 2400,
            "syllables": [
                {"text": "First", "start_ms": 900, "end_ms": 1300},
                {"text": " line", "start_ms": 1900, "end_ms": 2400},
            ],
        }
    ]
    source = [{"text": "First line", "start_ms": 1000}, {"text": "Second line", "start_ms": 2000}]

    result = bound_to_synced_lines(karaoke, source)

    assert result[0]["start_ms"] == 1000
    assert result[0]["end_ms"] == 2000
    assert result[0]["syllables"] == [
        {"text": "First", "start_ms": 1000, "end_ms": 1300},
        {"text": " line", "start_ms": 1900, "end_ms": 2000},
    ]


@pytest.mark.parametrize("separate", [None, True, False])
def test_worker_uses_cached_vocals_and_original_reference(monkeypatch, tmp_path, separate):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setenv("FA_KARA_VOCAL_SEPARATION", "false")
    get_settings.cache_clear()
    snapshot = (
        tmp_path
        / "hub"
        / f"models--{pipeline.DEFAULT_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / pipeline.DEFAULT_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    events = []
    original = b"original compressed full mix"
    waveform = np.array([0.25, -0.125], dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, -waveform, 16000, format="WAV", subtype="FLOAT")

    def vocals(data, check, before_separate):
        assert data == original
        assert events == []
        before_separate()
        assert events == ["stop"]
        events.append("vocals")
        return buffer.getvalue()

    monkeypatch.setattr(pipeline, "_stop_fa_kara_worker", lambda: events.append("stop"))
    monkeypatch.setattr(pipeline, "vocal_audio_bytes", vocals)

    def decode(data, check):
        assert data == original and callable(check)
        return waveform

    monkeypatch.setattr(pipeline, "vocal_reference_waveform", decode)
    monkeypatch.setattr(pipeline, "_fa_kara_worker", lambda *args: object())

    def job(worker, argv):
        assert "--separate_vocals" not in argv
        work = Path(argv[argv.index("--path_io") + 1])
        input_path = Path(argv[argv.index("--input_audio") + 1])
        if separate is False:
            assert events == []
            assert input_path.read_bytes() == original
            assert "--reference_audio" not in argv
        else:
            assert events == ["stop", "vocals"]
            reference = Path(argv[argv.index("--reference_audio") + 1])
            for path, expected in ((input_path, -waveform), (reference, waveform)):
                info = sf.info(path)
                assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "FLOAT")
                np.testing.assert_array_equal(sf.read(path)[0], expected)
        assert (work / "i.txt").read_text() == "Hello\n"
        assert json.loads((work / "timeline.json").read_text())[0]["start_ms"] == 100
        (work / "o.ass").write_text("")
        (work / "o_ruby.lrc").write_text("")
        (work / "o.alignment.json").write_text(
            json.dumps(
                {
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
                }
            )
        )
        return {"ok": True}

    monkeypatch.setattr(pipeline, "_run_worker_job", job)
    result = pipeline._run_fa_kara(
        original, "Hello", "en", [{"text": "Hello", "start_ms": 100}], separate
    )
    assert result["diagnostics"]["separator"] == (None if separate is False else "roformer")


@pytest.mark.parametrize("cancel", [False, True])
def test_backfill_prewarms_before_alignment_and_skips_failed_tracks(monkeypatch, cancel):
    monkeypatch.setenv("DATABASE_URL", "unused-mocked")
    get_settings.cache_clear()
    connection = MagicMock()
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda _: connection)
    connection.__enter__.return_value = connection
    tracks = [(i, str(i), str(i), "Hello", "en", []) for i in range(3)]
    connection.cursor.return_value.__enter__.return_value.fetchall.return_value = tracks
    client = MagicMock()
    client.__enter__.return_value = client
    client.audio_bytes.side_effect = lambda key: key.encode()
    monkeypatch.setattr(pipeline, "NavidromeClient", lambda *args: client)
    monkeypatch.setattr(pipeline, "resolve_library_id", lambda *args: "library")
    monkeypatch.setattr(
        pipeline,
        "plan_karaoke",
        lambda *args, **kwargs: SimpleNamespace(karaoke_external_ids=["0", "1", "2"]),
    )
    events = []
    monkeypatch.setattr(pipeline, "_stop_fa_kara_worker", lambda: events.append("stop"))

    def prepare(data, **kwargs):
        assert kwargs["reference"] is True
        assert "mono_rates" not in kwargs
        assert kwargs["vocals"] is True
        assert callable(kwargs["check"])
        events.append(data)
        if data == b"1":
            if cancel:
                raise KeyboardInterrupt()
            raise ValueError("bad audio")

    monkeypatch.setattr(pipeline, "prepare_audio", prepare)

    def align(data, *args):
        assert events[:4] == ["stop", b"0", b"1", b"2"]
        assert data != b"1"
        events.append(("align", data))
        return {
            "lines": [],
            "diagnostics": {
                "audio_source": "roformer_vocals",
                "reference_audio_source": "full_mix",
                "separator_revision": "revision",
            },
            "model": "model",
            "model_revision": "revision",
            "ass": "",
            "lrc": "",
            "alignment_document": {},
        }

    monkeypatch.setattr(pipeline, "_run_fa_kara", align)
    reports = []
    if cancel:
        with pytest.raises(KeyboardInterrupt):
            pipeline._backfill_karaoke("url", "user", "pass", reports.append)
        assert events == ["stop", b"0", b"1"]
    else:
        assert pipeline._backfill_karaoke("url", "user", "pass", reports.append) == {
            "total": 3,
            "aligned": 2,
            "failed": 1,
        }
        assert events[4:] == [("align", b"0"), ("align", b"2")]
        assert [r["phase"] for r in reports][:7] == ["preprocess"] * 6 + ["models"]
        # Later phases reread the batch's source cache rather than copy raw audio
        # into a second unbounded per-karaoke directory.
        assert client.audio_bytes.call_count == 5


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
            assert pipeline._run_worker_job(worker, []) == {"ok": True}
            assert checks >= 2
            worker.kill.assert_not_called()
        elif mode == "partial_cancel":
            os.write(writer, b'{"ok":')
            with pytest.raises(Cancelled):
                pipeline._run_worker_job(worker, [])
            worker.kill.assert_called_once()
            worker.wait.assert_called_once_with(timeout=5)
        else:
            with pytest.raises(TimeoutError):
                pipeline._run_worker_job(worker, [], timeout=0)
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

    monkeypatch.setattr(pipeline, "get_check", lambda: check)
    with pytest.raises(Cancelled):
        pipeline.backfill_karaoke("url", "user", "pass")
    lock.acquire.assert_called_once_with(timeout=0.25)
    lock.release.assert_not_called()


def test_local_model_path_overrides_snapshot(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint-22000"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights v1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "missing-cache"))
    monkeypatch.setenv("FA_KARA_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("FA_KARA_REVISION", "v6-checkpoint-22000")
    get_settings.cache_clear()
    monkeypatch.setattr(pipeline, "_fa_kara_worker", lambda *args: object())
    seen = {}

    def job(worker, argv):
        seen["model"] = argv[argv.index("--hf_model_path") + 1]
        work = Path(argv[argv.index("--path_io") + 1])
        (work / "o.ass").write_text("")
        (work / "o_ruby.lrc").write_text("")
        (work / "o.alignment.json").write_text(
            json.dumps(
                {
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
                }
            )
        )
        return {"ok": True}

    monkeypatch.setattr(pipeline, "_run_worker_job", job)
    try:
        result = pipeline._run_fa_kara(b"audio", "Hello", "en", [], False)
    finally:
        get_settings.cache_clear()
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
    get_settings.cache_clear()
    try:
        # Without a local checkpoint, the pinned revision is the identity.
        assert pipeline.current_model_revision() == "same-label"
        monkeypatch.setenv("FA_KARA_MODEL_PATH", str(checkpoint))
        get_settings.cache_clear()
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
        get_settings.cache_clear()


def test_second_stream_words_that_overlap_are_marked_as_a_second_voice():
    from echora_analysis.karaoke_pipeline import mark_backing_voice

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
            "syllables": [  # parenthesized, sung by the lead alone: stays on the line
                {"text": "Oh ", "start_ms": 4000, "end_ms": 4500},
                {"text": "(yeah) ", "start_ms": 4500, "end_ms": 5000},
                {"text": "yeah", "start_ms": 5000, "end_ms": 5500, "voice": 2},
            ],
        },
        {
            "text": "Call (me) now",
            "syllables": [  # on the second stream but sung after the lead: stays on the line
                {"text": "Call ", "start_ms": 6000, "end_ms": 6500},
                {"text": "(me) ", "start_ms": 6520, "end_ms": 6900, "stream": 2},
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
    # Stale mark cleared.
    assert [s.get("voice") for s in lines[1]["syllables"]] == [None, None, None]
    assert [s.get("voice") for s in lines[2]["syllables"]] == [None, None, None]
    assert [s.get("voice") for s in lines[3]["syllables"]] == [None, None]


def test_overlap_between_lines_is_not_a_second_voice():
    from echora_analysis.karaoke_pipeline import mark_backing_voice

    lines = [
        {"text": "First", "syllables": [{"text": "First ", "start_ms": 0, "end_ms": 2000}]},
        {
            "text": "Second",
            "syllables": [{"text": "Second ", "start_ms": 1500, "end_ms": 2500, "stream": 2}],
        },
    ]
    mark_backing_voice(lines)
    assert all("voice" not in s for line in lines for s in line["syllables"])
