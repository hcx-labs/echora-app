"""Pre-batch song selection, separate from execution-time rechecking."""

from echora_analysis.settings import get_settings

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from echora_analysis import sync_plan


@pytest.fixture
def planner(monkeypatch):
    from echora_analysis import translation_storage

    # Optional pipelines add their own selection queries; keep the host's config out.
    for name in ("MOSS_MODEL_ID", "MOSS_REVISION", "ECHORA_RECORDING_MODEL_MANIFEST"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    monkeypatch.setattr(
        translation_storage, "load_settings", lambda: (SimpleNamespace(enabled=False), None)
    )
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (uuid4(),)
    audio = Mock(return_value=SimpleNamespace(download_external_ids=set()))
    karaoke = Mock(return_value=SimpleNamespace(karaoke_external_ids=()))
    monkeypatch.setattr(sync_plan, "plan_audio", audio)
    monkeypatch.setattr(sync_plan, "plan_karaoke", karaoke)
    monkeypatch.setattr(sync_plan, "sources_needing_refresh", Mock(return_value=set()))
    monkeypatch.setattr(sync_plan, "sync_selection", Mock(return_value=set()))
    return connection, cursor, audio, karaoke


def test_entire_library_visits_songs_whose_album_needs_motion_artwork(planner, monkeypatch):
    connection, cursor, _, _ = planner
    cursor.fetchall.side_effect = [[("a",), ("b",)], [], []]
    artwork = Mock(return_value={"b"})
    monkeypatch.setattr(sync_plan, "sync_selection", artwork)
    assert sync_plan.select_sync_tracks(connection, "https://music", ["a", "b"]) == ["b"]
    assert artwork.call_args.args[1:] == (cursor.fetchone.return_value[0], ["a", "b"])


def test_new_only_excludes_existing_even_if_they_need_backfill(planner):
    connection, cursor, audio, karaoke = planner
    existing = [f"old-{i}" for i in range(936)]
    new = [f"new-{i}" for i in range(5)]
    cursor.fetchall.return_value = [(song,) for song in existing]
    assert (
        sync_plan.select_sync_tracks(connection, "https://music", existing + new, "missing") == new
    )
    audio.assert_not_called()
    karaoke.assert_not_called()


def test_karaoke_revision_backfill_combines_existing_and_new(planner):
    connection, cursor, audio, karaoke = planner
    old = [f"old-{i}" for i in range(100)]
    new = [f"new-{i}" for i in range(10)]
    cursor.fetchall.side_effect = [[(song,) for song in old], []]
    karaoke.return_value.karaoke_external_ids = tuple(old)
    audio.return_value.download_external_ids = set(new)
    assert sync_plan.select_sync_tracks(connection, "https://music", old + new) == old + new
    assert karaoke.call_args.kwargs["library_id"] == cursor.fetchone.return_value[0]


def test_current_catalog_creates_no_work(planner):
    connection, cursor, _, _ = planner
    cursor.fetchall.side_effect = [[("current",)], []]
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
def test_parent_batches_selected_work_but_reconciles_full_catalog(monkeypatch, recording_enabled):
    import echora_analysis
    from echora_analysis import analysis_jobs

    catalog = [f"song-{i}" for i in range(941)]
    selected = catalog[-70:]
    if recording_enabled:
        monkeypatch.setenv("ECHORA_RECORDING_MODEL_MANIFEST", "/configured/model.json")
        get_settings.cache_clear()
    else:
        monkeypatch.delenv("ECHORA_RECORDING_MODEL_MANIFEST", raising=False)
        get_settings.cache_clear()
    main = SimpleNamespace(
        _load_connection=Mock(return_value=("https://music", "user", "secret")),
        _attach_user_library=Mock(),
        _reconcile_user_tracks=Mock(),
    )
    store = SimpleNamespace(expand=Mock(return_value=True))
    monkeypatch.setattr(echora_analysis, "main", main, raising=False)
    monkeypatch.setattr(echora_analysis, "jobs", store, raising=False)
    client = MagicMock()
    client.__enter__.return_value.all_tracks.return_value = [
        SimpleNamespace(id=song) for song in catalog
    ]
    monkeypatch.setattr(analysis_jobs, "NavidromeClient", lambda *args: client)

    @contextmanager
    def connect():
        yield "db"

    monkeypatch.setattr(analysis_jobs, "_connect", connect)
    select = Mock(return_value=selected)
    monkeypatch.setattr(sync_plan, "select_sync_tracks", select)
    monkeypatch.setenv("ECHORA_BATCH_SIZE", "32")
    get_settings.cache_clear()
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
    from echora_analysis.processing_plan import AudioProcessingPlan

    monkeypatch.delenv("ECHORA_RECORDING_MATCH_POLICY", raising=False)
    get_settings.cache_clear()
    monkeypatch.delenv("MOSS_MODEL_ID", raising=False)
    monkeypatch.delenv("MOSS_REVISION", raising=False)
    get_settings.cache_clear()
    connection, cursor, audio, _ = planner
    cursor.fetchall.side_effect = [[("existing",), ("ready",)], []]
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
    cursor.fetchall.return_value = [("known",)]
    assert sync_plan.select_sync_tracks(
        connection, "https://music", ["known", "new"], verify_audio_hashes=False
    ) == ["known", "new"]
    audio.assert_not_called()
    karaoke.assert_not_called()
