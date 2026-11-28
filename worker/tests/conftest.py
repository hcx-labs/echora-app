"""Keep cached process settings from leaking between tests."""

import pytest

from core.config import reset_config_cache
from core.testing import database  # noqa: F401 - shared fixture


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_compute_choice: use the database-backed compute choice"
    )
    config.addinivalue_line("markers", "real_resync: use the database's resync requests")


@pytest.fixture(autouse=True)
def reset_settings_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture(autouse=True)
def local_compute_by_default(monkeypatch, request):
    """Jobs without a recorded location ask the database for the instance default.

    Unit tests have no database, so they compute locally unless a test opts out.
    """
    if "real_compute_choice" in request.keywords:
        yield
        return
    import worker.remote.settings as external_processing

    monkeypatch.setattr(
        external_processing, "compute_for", lambda user_id, requested: requested or "local"
    )
    yield


@pytest.fixture(autouse=True)
def no_resync_requests(monkeypatch, request):
    """Songs have no resync requests unless a test says so.

    Unit tests queue their database results; database-backed tests are marked ``real_resync``.
    """
    if "real_resync" in request.keywords:
        yield
        return
    from core.resync import store as resync

    monkeypatch.setattr(resync, "requested", lambda db, pipeline, track_ids: {})
    monkeypatch.setattr(resync, "requested_sources", lambda db, library_id, ids: set())
    monkeypatch.setattr(resync, "done", lambda db, track_id, pipeline: None)
    yield


# Helpers the worker once reached through the API module, by their old and current names.
_MOVED_HELPERS = {
    "_load_connection": "load_connection",
    "_reconcile_user_tracks": "reconcile_user_tracks",
    "_attach_user_library": "attach_user_library",
    "_relink_renumbered_sources": "relink_renumbered_sources",
    "_user_audio_track_ids": "user_audio_track_ids",
    "_cipher": "credential_cipher",
    "CurationPreviewRequest": "CurationRecipe",
    "_preview_curation": "preview_curation",
    "NavidromeClient": "NavidromeClient",
}


@pytest.fixture
def library_helpers(monkeypatch):
    """Replace core helpers (connections, membership, curation preview) in every worker module.

    Call it with an object whose attributes are the fakes, under either name.
    """
    import sys

    def install(fakes):
        for old, new in _MOVED_HELPERS.items():
            for name in (old, new):
                if not hasattr(fakes, name):
                    continue
                for module in list(sys.modules.values()):
                    if getattr(module, "__name__", "").startswith("worker.") and hasattr(
                        module, new
                    ):
                        monkeypatch.setattr(module, new, getattr(fakes, name))

    return install
