import backend.api.v1.library.top_tracks as endpoint


class Preference:
    lastfm_username = None
    lastfm_api_key_encrypted = None


class Session:
    def __init__(self, preference):
        self.preference = preference

    def get(self, model, key):
        return self.preference

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_unknown_period_is_422(client):
    assert client.get("/api/v1/library/top-tracks?period=forever").status_code == 422


def test_without_lastfm_or_navidrome_nothing_is_available(client, monkeypatch):
    monkeypatch.setattr(endpoint, "session_scope", lambda: Session(None))
    assert client.get("/api/v1/library/top-tracks").json() == {
        "available": False,
        "source": None,
        "period": "1month",
        "tracks": [],
    }


def test_navidrome_counts_are_all_time(client, user, monkeypatch):
    user["navidrome_connection_id"] = "c1"
    monkeypatch.setattr(endpoint, "session_scope", lambda: Session(Preference()))
    monkeypatch.setattr(endpoint.ranking, "navidrome_ranking", lambda user_id, conn: [("s1", 4)])
    monkeypatch.setattr(
        endpoint.ranking,
        "library_track_rows",
        lambda user_id: [{"id": "t1", "title": "One", "source_id": "s1"}],
    )
    body = client.get("/api/v1/library/top-tracks").json()
    assert body["source"] == "navidrome" and body["period"] == "overall"
    assert body["tracks"] == [{"id": "t1", "title": "One", "source_id": "s1", "play_count": 4}]


def test_navidrome_failure_is_502(client, user, monkeypatch):
    user["navidrome_connection_id"] = "c1"
    monkeypatch.setattr(endpoint, "session_scope", lambda: Session(None))

    def fail(user_id, connection_id):
        raise RuntimeError("down")

    monkeypatch.setattr(endpoint.ranking, "navidrome_ranking", fail)
    response = client.get("/api/v1/library/top-tracks")
    assert response.status_code == 502
    assert response.json() == {"detail": "Could not read Navidrome play counts: down"}
