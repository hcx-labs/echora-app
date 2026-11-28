"""Pre-batch song selection, separate from execution-time rechecking."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

import worker.planning.sync as sync_plan
from core.config import reset_config_cache
from core.testing import stub_module


@pytest.fixture
def planner(monkeypatch):
    import core.lyrics.translations as translation_storage

    # Optional pipelines add their own selection queries; keep the host's config out.
    for name in ("MOSS_MODEL_ID", "MOSS_REVISION", "ECHORA_RECORDING_MODEL_MANIFEST"):
        monkeypatch.delenv(name, raising=False)
    reset_config_cache()

    monkeypatch.setattr(
        translation_storage, "load_settings", lambda: (SimpleNamespace(enabled=False), None)
    )
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = {"id": uuid4()}
    audio = Mock(return_value=SimpleNamespace(download_external_ids=set()))
    karaoke = Mock(return_value=SimpleNamespace(karaoke_external_ids=()))
    monkeypatch.setattr(sync_plan, "plan_audio", audio)
    monkeypatch.setattr(sync_plan, "plan_karaoke", karaoke)
    monkeypatch.setattr(sync_plan, "sources_needing_refresh", Mock(return_value=set()))
    monkeypatch.setattr(sync_plan, "sync_selection", Mock(return_value=set()))
    monkeypatch.setattr(sync_plan, "songs_needing_stems", Mock(return_value=set()))
    return connection, cursor, audio, karaoke


def test_entire_library_visits_songs_whose_album_needs_motion_artwork(planner, monkeypatch):
    connection, cursor, _, _ = planner
    cursor.fetchall.side_effect = [[{"external_id": "a"}, {"external_id": "b"}], [], []]
    artwork = Mock(return_value={"b"})
    monkeypatch.setattr(sync_plan, "sync_selection", artwork)
    assert sync_plan.select_sync_tracks(connection, "https://music", ["a", "b"]) == ["b"]
    assert artwork.call_args.args[1:] == (cursor.fetchone.return_value["id"], ["a", "b"])


def test_new_only_excludes_existing_even_if_they_need_backfill(planner):
    connection, cursor, audio, karaoke = planner
    existing = [f"old-{i}" for i in range(936)]
    new = [f"new-{i}" for i in range(5)]
    cursor.fetchall.return_value = [{"external_id": song} for song in existing]
    assert (
        sync_plan.select_sync_tracks(connection, "https://music", existing + new, "missing") == new
    )
    audio.assert_not_called()
    karaoke.assert_not_called()


def test_karaoke_revision_backfill_combines_existing_and_new(planner):
    connection, cursor, audio, karaoke = planner
    old = [f"old-{i}" for i in range(100)]
    new = [f"new-{i}" for i in range(10)]
    cursor.fetchall.side_effect = [[{"external_id": song} for song in old], []]
    karaoke.return_value.karaoke_external_ids = tuple(old)
    audio.return_value.download_external_ids = set(new)
    assert sync_plan.select_sync_tracks(connection, "https://music", old + new) == old + new
    assert karaoke.call_args.kwargs["library_id"] == cursor.fetchone.return_value["id"]


def test_current_catalog_creates_no_work(planner):
    connection, cursor, _, _ = planner
    cursor.fetchall.side_effect = [[{"external_id": "current"}], []]
    assert sync_plan.select_sync_tracks(connection, "https://music", ["current"]) == []


def test_new_library_selects_every_song_once(planner):
    connection, cursor, audio, _ = planner
    cursor.fetchone.return_value = None
    assert sync_plan.select_sync_tracks(connection, "https://music", ["new", "new"]) == ["new"]
    audio.assert_not_called()


def test_empty_catalog_and_invalid_mode(planner):
    connection, _, _, _ = planner
    assert sync_plan.select_sync_tracks(connection, "https://music", []) == []
    with pytest.raises(ValueError):
        sync_plan.select_sync_tracks(connection, "https://music", [], "force")


@pytest.mark.parametrize("recording_enabled", [False, True])
def test_parent_batches_selected_work_but_reconciles_full_catalog(
    monkeypatch, recording_enabled, library_helpers
):
    import worker.operations as analysis_jobs

    catalog = [f"song-{i}" for i in range(941)]
    selected = catalog[-70:]
    if recording_enabled:
        monkeypatch.setenv("ECHORA_RECORDING_MODEL_MANIFEST", "/configured/model.json")
        reset_config_cache()
    else:
        monkeypatch.delenv("ECHORA_RECORDING_MODEL_MANIFEST", raising=False)
        reset_config_cache()
    main = SimpleNamespace(
        _load_connection=Mock(return_value=("https://music", "user", "secret")),
        _attach_user_library=Mock(),
        _reconcile_user_tracks=Mock(),
        _relink_renumbered_sources=Mock(return_value=0),
    )
    store = SimpleNamespace(expand=Mock(return_value=True))
    stub_module(monkeypatch, "core.jobs.queue", store)
    library_helpers(main)
    client = MagicMock()
    client.__enter__.return_value.all_tracks.return_value = [
        SimpleNamespace(id=song) for song in catalog
    ]
    monkeypatch.setattr(analysis_jobs, "NavidromeClient", lambda *args: client)

    @contextmanager
    def connect(**_):
        yield "db"

    monkeypatch.setattr(analysis_jobs, "connect", connect)
    select = Mock(return_value=selected)
    monkeypatch.setattr(sync_plan, "select_sync_tracks", select)
    monkeypatch.setenv("ECHORA_BATCH_SIZE", "32")
    reset_config_cache()
    context = SimpleNamespace(check=Mock(), report=Mock(), token="token")
    job = {
        "id": "root",
        "kind": "navidrome_sync",
        "user_id": "user",
        "connection_id": "connection",
        "payload": {"mode": "missing"},
    }
    assert analysis_jobs.execute(job, context) is None
    select.assert_called_once_with(
        "db",
        "https://music",
        catalog,
        "missing",
        catalog=client.__enter__.return_value.all_tracks.return_value,
        verify_audio_hashes=True,
    )
    main._reconcile_user_tracks.assert_called_once_with("user", "https://music", catalog)
    assert store.expand.call_args.args[2] == [
        {
            "operation": "navidrome_sync",
            "connection_id": "connection",
            "compute": "local",
            "track_ids": selected[i : i + 32],
            "verify_audio_hashes": True,
            "refresh_lyrics": False,
        }
        for i in range(0, len(selected), 32)
    ]


def test_entire_library_selects_recording_only_backfill_without_other_models(planner, monkeypatch):
    from worker.planning.audio import AudioProcessingPlan

    monkeypatch.delenv("ECHORA_RECORDING_MATCH_POLICY", raising=False)
    reset_config_cache()
    monkeypatch.delenv("MOSS_MODEL_ID", raising=False)
    monkeypatch.delenv("MOSS_REVISION", raising=False)
    reset_config_cache()
    connection, cursor, audio, _ = planner
    cursor.fetchall.side_effect = [[{"external_id": "existing"}, {"external_id": "ready"}], []]
    audio.return_value = AudioProcessingPlan(
        frozenset(),
        frozenset(),
        frozenset(),
        frozenset(),
        recording_fingerprint_external_ids=frozenset({"existing"}),
    )
    assert sync_plan.select_sync_tracks(
        connection, "https://music", ["existing", "ready"], "all"
    ) == ["existing"]
    assert not audio.return_value.needs_muq and not audio.return_value.needs_mert


def test_skip_hash_visits_known_tracks_for_metadata_and_translation(planner):
    connection, cursor, audio, karaoke = planner
    cursor.fetchall.return_value = [{"external_id": "known"}]
    assert sync_plan.select_sync_tracks(
        connection, "https://music", ["known", "new"], verify_audio_hashes=False
    ) == ["known", "new"]
    audio.assert_not_called()
    karaoke.assert_not_called()


def test_entire_library_visits_songs_without_karaoke_stems(planner, monkeypatch):
    connection, cursor, _, _ = planner
    cursor.fetchall.side_effect = [[{"external_id": "a"}, {"external_id": "b"}], [], []]
    stems = Mock(return_value={"a"})
    monkeypatch.setattr(sync_plan, "songs_needing_stems", stems)
    assert sync_plan.select_sync_tracks(connection, "https://music", ["a", "b"]) == ["a"]
    assert stems.call_args.args[2] == ["a", "b"]


def test_songs_asked_to_resync_join_a_partial_sync(planner, monkeypatch):
    from core.resync import store as resync

    connection, cursor, audio, karaoke = planner
    cursor.fetchall.return_value = [{"external_id": song} for song in ("known", "asked")]
    monkeypatch.setattr(resync, "requested_sources", lambda db, library, ids: {"asked"})
    assert sync_plan.select_sync_tracks(
        connection, "https://music", ["known", "asked", "new"], "missing"
    ) == ["asked", "new"]
    karaoke.assert_not_called()
