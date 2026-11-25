"""The onboarding steps: a random sample from the saved connection, then importing the picks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import backend.api.v1.sync.sample as sample
import backend.sync.connections as connections
import core.jobs.queue as jobs

from ._fakes import snapshot

OWNED = "00000000-0000-0000-0000-00000000c0de"
FOREIGN = "00000000-0000-0000-0000-00000000f0e1"
CREDENTIALS = ("http://navidrome", "me", "secret")


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
def owned(monkeypatch, user):
    def load(connection_id, user_id):
        assert user_id == user["id"]
        return CREDENTIALS if connection_id == OWNED else None

    monkeypatch.setattr(connections, "load_connection", load)


def test_sample_reads_random_tracks_from_the_saved_connection(client, owned, monkeypatch):
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

    monkeypatch.setattr(sample, "NavidromeClient", Client)
    result = client.get(f"/api/v1/sync/connections/{OWNED}/sample?limit=24").json()
    assert seen == {"credentials": CREDENTIALS, "limit": 24}
    assert [t["id"] for t in result["tracks"]] == ["t1", "t2"]
    assert result["tracks"][0]["cover_art"] == "al-1"
    assert client.get(f"/api/v1/sync/connections/{FOREIGN}/sample").status_code == 404


def test_import_queues_the_chosen_tracks_without_credentials(client, user, owned, monkeypatch):
    queue = Mock(side_effect=lambda kind, *a, **k: snapshot(kind))
    monkeypatch.setattr(jobs, "enqueue", queue)
    body = {"track_ids": ["b", "a", "b"]}
    assert client.post(f"/api/v1/sync/connections/{OWNED}/import", json=body).status_code == 202
    ((args, kwargs),) = [(c.args, c.kwargs) for c in queue.call_args_list]
    assert args == ("import", "analysis", user["id"])
    assert kwargs["connection_id"] == OWNED
    assert kwargs["payload"] == {"track_ids": ["a", "b"]}
    assert "secret" not in repr(kwargs)
    response = client.post(f"/api/v1/sync/connections/{FOREIGN}/import", json=body)
    assert response.status_code == 404
    assert queue.call_count == 1
