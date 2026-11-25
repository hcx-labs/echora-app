"""The plugin forwards the Navidrome PDK's request structs, which carry fields Echora ignores."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.search.plugin as ranking
from backend.api.errors import install_error_handlers
from backend.api.v1.plugin import CATEGORY
from backend.settings.plugin_integration import RankingProfile, authenticate


@pytest.fixture
def plugin(monkeypatch):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(CATEGORY.router(), prefix="/api/v1")
    principal = {"user_id": "u", "url": "http://navidrome", "profile": RankingProfile()}
    app.dependency_overrides[authenticate] = lambda: principal
    calls = []
    monkeypatch.setattr(ranking, "load_corpus", lambda p: "corpus")
    monkeypatch.setattr(ranking, "similar_tracks", lambda *a: calls.append(a) or {"matches": []})
    monkeypatch.setattr(ranking, "sonic_path", lambda *a: calls.append(a) or {"matches": []})
    monkeypatch.setattr(ranking, "similar_artists", lambda *a: calls.append(a) or {"artists": []})
    return TestClient(app), calls


def test_unknown_fields_are_ignored_at_top_level_and_nested(plugin):
    client, calls = plugin
    song = {"id": "s1", "name": "Song", "artist": "Artist", "mbid": "x", "durationMs": 1}
    response = client.post(
        "/api/v1/plugin/similar-tracks", json={"song": song, "count": 5, "extra": {"a": 1}}
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/v1/plugin/sonic-path",
        json={"startSong": song, "endSong": {**song, "id": "s2"}, "count": 3, "unknown": True},
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/v1/plugin/similar-artists",
        json={"id": "a1", "name": "Artist", "mbid": "m", "limit": 4},
    )
    assert response.status_code == 200, response.text
    assert [call[1:4] for call in calls] == [
        ("s1", 5, calls[0][3]),
        ("s1", "s2", 3),
        ("a1", 4, calls[2][3]),
    ]


def test_known_fields_are_still_validated(plugin):
    client, _ = plugin
    assert (
        client.post("/api/v1/plugin/similar-tracks", json={"song": {"id": ""}}).status_code == 422
    )
    assert (
        client.post(
            "/api/v1/plugin/similar-tracks", json={"song": {"id": "s"}, "count": 10000}
        ).status_code
        == 422
    )
