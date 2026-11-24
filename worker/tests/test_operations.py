"""Orchestration tests use no database or model downloads."""

from core.testing import stub_module
from core.config import reset_config_cache

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import worker.operations as analysis_jobs


class Cancelled(BaseException):
    pass


def context():
    return SimpleNamespace(check=Mock(), report=Mock(), token="lease")


def test_import_expands_only_selected_ids_without_reconciliation(monkeypatch, library_helpers):
    main = SimpleNamespace(
        _load_connection=Mock(return_value=("https://music", "u", "secret")),
        _reconcile_user_tracks=Mock(),
    )
    jobs = SimpleNamespace(expand=Mock())
    library_helpers(main)
    stub_module(monkeypatch, "core.jobs.queue", jobs)
    monkeypatch.setenv("ECHORA_BATCH_SIZE", "2")
    reset_config_cache()
    job = {
        "id": "parent",
        "kind": "import",
        "user_id": "user",
        "payload": {
            "connection_id": "connection",
            "track_ids": ["a", "b", "a", "c"],
            "password": "DO NOT COPY",
        },
    }
    assert analysis_jobs.execute(job, context()) is None
    assert jobs.expand.call_args.args == (
        "parent",
        "lease",
        [
            {
                "operation": "import",
                "connection_id": "connection",
                "compute": "local",
                "track_ids": ["a", "b"],
            },
            {
                "operation": "import",
                "connection_id": "connection",
                "compute": "local",
                "track_ids": ["c"],
            },
        ],
    )
    main._reconcile_user_tracks.assert_not_called()


def test_recording_search_uses_same_worker_without_loading_connection(monkeypatch):
    import worker.recordings.search as recording_search

    execute = Mock(return_value={"state": "no_match"})
    monkeypatch.setattr(recording_search, "execute", execute)
    job = {"kind": "recording_search", "id": "query", "user_id": "user"}
    ctx = context()
    assert analysis_jobs.execute(job, ctx) == {"state": "no_match"}
    execute.assert_called_once_with(job, ctx)


def test_recording_model_does_not_override_configured_batch_size(monkeypatch, library_helpers):
    main = SimpleNamespace(_load_connection=Mock(return_value=("https://music", "u", "secret")))
    queue = SimpleNamespace(expand=Mock())
    library_helpers(main)
    stub_module(monkeypatch, "core.jobs.queue", queue)
    monkeypatch.setenv("ECHORA_BATCH_SIZE", "128")
    reset_config_cache()
    monkeypatch.setenv("ECHORA_RECORDING_MODEL_MANIFEST", "/local/manifest.json")
    reset_config_cache()
    job = {
        "id": "parent",
        "kind": "import",
        "user_id": "user",
        "payload": {"connection_id": "connection", "track_ids": ["a", "b"]},
    }
    analysis_jobs.execute(job, context())
    assert [batch["track_ids"] for batch in queue.expand.call_args.args[2]] == [["a", "b"]]


def test_failed_scan_never_reconciles_or_expands(monkeypatch, library_helpers):
    main = SimpleNamespace(
        _load_connection=Mock(return_value=("url", "u", "p")),
        _attach_user_library=Mock(),
        _reconcile_user_tracks=Mock(),
    )
    jobs = SimpleNamespace(expand=Mock())
    library_helpers(main)
    stub_module(monkeypatch, "core.jobs.queue", jobs)
    with patch.object(analysis_jobs, "NavidromeClient") as client:
        client.return_value.__enter__.return_value.all_tracks.side_effect = RuntimeError("partial")
        with pytest.raises(RuntimeError):
            analysis_jobs.execute(
                {
                    "id": "p",
                    "kind": "navidrome_sync",
                    "user_id": "u",
                    "payload": {"connection_id": "c"},
                },
                context(),
            )
    main._reconcile_user_tracks.assert_not_called()
    jobs.expand.assert_not_called()


def test_cancellation_propagates_before_imports():
    ctx = context()
    ctx.check.side_effect = Cancelled()
    with pytest.raises(Cancelled):
        analysis_jobs.execute({}, ctx)


def test_batch_profiles_are_canonical_and_scoped(monkeypatch, library_helpers):
    main = SimpleNamespace(_user_audio_track_ids=Mock(return_value=["owned", "other"]))
    library_helpers(main)
    with patch.object(analysis_jobs, "_profiles", return_value={"profiled": 1}) as profiles:
        result = analysis_jobs.execute(
            {
                "kind": "analysis_batch",
                "user_id": "u",
                "payload": {"operation": "audio_profiles", "track_ids": ["owned", "foreign"]},
            },
            context(),
        )
    assert result == {"profiled": 1}
    assert profiles.call_args.args[0] == ["owned"]


def test_hum_parent_creates_durable_corpus_from_job_id(monkeypatch, library_helpers):
    from unittest.mock import MagicMock

    main = SimpleNamespace(_load_connection=Mock(return_value=("url", "u", "p")))
    jobs = SimpleNamespace(expand=Mock())
    library_helpers(main)
    stub_module(monkeypatch, "core.jobs.queue", jobs)
    hum = SimpleNamespace(DEFAULT_CORPUS_SIZE=100)
    stub_module(monkeypatch, "worker.melody.index", hum)
    db = MagicMock()
    cursor = db.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ("user",)
    monkeypatch.setattr(analysis_jobs, "_connect", lambda: db)
    monkeypatch.setattr(
        analysis_jobs, "_source_rows", lambda *a: [("a", "canonical"), ("b", "other")]
    )
    job = {
        "id": "parent",
        "kind": "hum_corpus",
        "user_id": "user",
        "connection_id": "connection",
        "payload": {"track_limit": 1},
    }
    assert analysis_jobs.execute(job, context()) is None
    assert cursor.execute.call_args_list[0].args[1] == ("parent", "user", 1)
    assert jobs.expand.call_args.args[2] == [
        {
            "operation": "hum_corpus",
            "connection_id": "connection",
            "compute": "local",
            "corpus_id": "parent",
            "track_ids": ["a"],
        }
    ]
    main._load_connection.assert_called_once_with("connection", "user")
