from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import echora_analysis.main as main


def track(i):
    return SimpleNamespace(
        id=f"t{i}",
        title=f"Song {i}",
        artist="Artist",
        album="Album",
        duration=180,
        genre=None,
        raw={"coverArt": f"al-{i}"},
    )


@pytest.fixture
def signed_in(monkeypatch):
    monkeypatch.setattr(main, "_session_user", lambda session: {"id": "user"})
    owned = {"conn": ("http://navidrome", "me", "secret")}
    monkeypatch.setattr(main, "load_connection", lambda cid, uid: owned.get(cid))


def test_sample_reads_random_tracks_from_the_saved_connection(signed_in, monkeypatch):
    seen = {}

    class Client:
        def __init__(self, *credentials):
            seen["credentials"] = credentials

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def random_tracks(self, limit):
            seen["limit"] = limit
            return [track(1), track(2)]

    monkeypatch.setattr(main, "NavidromeClient", Client)
    result = main.navidrome_sample("conn", limit=24, echora_session="s")
    assert seen == {"credentials": ("http://navidrome", "me", "secret"), "limit": 24}
    assert [t["id"] for t in result["tracks"]] == ["t1", "t2"]
    assert result["tracks"][0]["cover_art"] == "al-1"
    with pytest.raises(HTTPException) as missing:
        main.navidrome_sample("someone-elses", limit=24, echora_session="s")
    assert missing.value.status_code == 404


def test_import_queues_the_chosen_tracks_without_credentials(signed_in, monkeypatch):
    queued = []
    monkeypatch.setattr(
        main.jobs,
        "enqueue",
        lambda *args, **kwargs: queued.append((args, kwargs)) or {"job_id": "j"},
    )
    request = main.SampleImportRequest(track_ids=["b", "a", "b"])
    assert main.start_connection_import("conn", request, echora_session="s") == {"job_id": "j"}
    ((args, kwargs),) = queued
    assert args == ("import", "analysis", "user")
    assert kwargs["connection_id"] == "conn"
    assert kwargs["payload"] == {"track_ids": ["a", "b"]}
    assert "secret" not in repr(kwargs)
    with pytest.raises(HTTPException) as missing:
        main.start_connection_import("someone-elses", request, echora_session="s")
    assert missing.value.status_code == 404
