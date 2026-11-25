from unittest.mock import Mock
from uuid import uuid4

import pytest

import backend.api.v1.media.cover as cover
import backend.api.v1.media.scrobble as scrobble
import backend.api.v1.media.stream as stream
import backend.sync.connections as connections
from backend.media.cache import CachedMedia

CREDENTIALS = ("https://music", "user", "secret")


@pytest.fixture(autouse=True)
def owned(monkeypatch):
    monkeypatch.setattr(connections, "load_connection", Mock(return_value=CREDENTIALS))


class FakeCache:
    def __init__(self, content=None):
        self.content = content
        self.stored = {}

    def get(self, key):
        return CachedMedia(self.content, "audio/mpeg") if self.content is not None else None

    def set(self, key, content, content_type):
        self.stored[key] = (content, content_type)


def test_stream_rejects_unknown_quality(client):
    response = client.get(f"/api/v1/media/connections/{uuid4()}/stream/s1?quality=999")
    assert response.status_code == 422
    assert response.json() == {"detail": "Quality must be original, 320, or 120"}


def test_cached_stream_answers_ranges(client, monkeypatch):
    monkeypatch.setattr(stream, "media_cache", lambda: FakeCache(b"0123456789"))
    url = f"/api/v1/media/connections/{uuid4()}/stream/s1?cache=player"
    response = client.get(url, headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    suffix = client.get(url, headers={"Range": "bytes=-3"})
    assert suffix.content == b"789"
    assert client.get(url, headers={"Range": "bytes=20-"}).status_code == 416
    whole = client.get(url)
    assert whole.status_code == 200
    assert whole.content == b"0123456789"
    assert whole.headers["cache-control"] == "private, max-age=3600"


class FakeUpstream:
    status_code = 206
    headers = {"content-type": "audio/flac", "content-range": "bytes 0-3/4", "x-other": "1"}

    def iter_raw(self, size):
        yield b"ab"
        yield b"cd"

    def close(self):
        pass


def test_uncached_stream_forwards_range_headers(client, monkeypatch):
    navidrome = Mock()
    navidrome.open_transcode_stream.return_value = FakeUpstream()
    monkeypatch.setattr(stream, "media_navidrome_client", lambda *c: navidrome)
    response = client.get(
        f"/api/v1/media/connections/{uuid4()}/stream/s1?quality=320",
        headers={"Range": "bytes=0-3"},
    )
    assert response.status_code == 206
    assert response.content == b"abcd"
    assert response.headers["content-range"] == "bytes 0-3/4"
    assert "x-other" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    navidrome.open_transcode_stream.assert_called_once_with("s1", 320, "bytes=0-3")


def test_cover_is_cached_and_bounded(client, monkeypatch):
    cache = FakeCache()
    monkeypatch.setattr(cover, "media_cache", lambda: cache)
    navidrome = Mock()
    navidrome.__enter__ = Mock(return_value=navidrome)
    navidrome.__exit__ = Mock(return_value=False)
    navidrome.cover_art.return_value = (b"png", "image/png")
    monkeypatch.setattr(cover, "media_navidrome_client", lambda *c: navidrome)
    response = client.get(f"/api/v1/media/connections/{uuid4()}/cover/al-1/x?size=5000")
    assert response.content == b"png"
    assert response.headers["content-type"] == "image/png"
    navidrome.cover_art.assert_called_once_with("al-1/x", 1600)
    assert list(cache.stored.values()) == [(b"png", "image/png")]


def test_scrobble_failure_is_a_bare_502(client, monkeypatch):
    monkeypatch.setattr(scrobble, "media_navidrome_client", Mock(side_effect=OSError("down")))
    body = {"connection_id": "c", "song_id": "s"}
    response = client.post("/api/v1/media/scrobble", json=body)
    assert response.status_code == 502
    assert response.json() == {"detail": "Scrobble failed"}


def test_scrobble_is_204(client, monkeypatch):
    navidrome = Mock()
    navidrome.__enter__ = Mock(return_value=navidrome)
    navidrome.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(scrobble, "media_navidrome_client", lambda *c: navidrome)
    body = {"connection_id": "c", "song_id": "s", "submission": False}
    response = client.post("/api/v1/media/scrobble", json=body)
    assert response.status_code == 204
    assert response.content == b""
    navidrome.scrobble.assert_called_once_with("s", False)


def test_motion_artwork_of_another_library_is_404(client, monkeypatch):
    import backend.api.v1.media.motion_artwork as motion

    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    cursor = connection.cursor.return_value
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)
    cursor.fetchone.return_value = None
    monkeypatch.setattr(motion, "connect", lambda **_: connection)
    response = client.get(f"/api/v1/media/motion-artwork/{uuid4()}.mp4")
    assert response.status_code == 404
    assert response.json() == {"detail": "Motion artwork not found"}
