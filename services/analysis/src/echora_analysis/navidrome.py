from __future__ import annotations

from .settings import get_settings

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass
import hashlib
import secrets
from urllib.parse import urljoin

import httpx


@dataclass(frozen=True)
class NavidromeTrack:
    id: str
    title: str
    artist: str | None
    album: str | None
    duration: float
    year: int | None
    genre: str | None
    suffix: str | None
    path: str | None
    raw: dict[str, object]


_media_http_client = httpx.Client(
    timeout=httpx.Timeout(
        get_settings().navidrome_timeout_seconds, read=get_settings().navidrome_read_timeout_seconds
    ),
    follow_redirects=True,
    limits=httpx.Limits(
        max_connections=get_settings().navidrome_max_connections,
        max_keepalive_connections=get_settings().navidrome_max_keepalive_connections,
        keepalive_expiry=get_settings().navidrome_keepalive_expiry_seconds,
    ),
)


# Context-local: concurrent workers never share authenticated audio or cancellation.
_batch_state: ContextVar = ContextVar("navidrome_batch", default=None)


@contextmanager
def batch_audio_cache(check=lambda: None):
    """Bounded private disk cache; removed even on BaseException cancellation."""
    with TemporaryDirectory(prefix="echora-audio-") as directory:
        state = {
            "directory": Path(directory),
            "size": 0,
            "limit": max(0, int(get_settings().audio_cache_bytes)),
            "check": check,
        }
        token = _batch_state.set(state)
        try:
            yield
        finally:
            _batch_state.reset(token)


class NavidromeClient:
    """Small Subsonic client. Credentials are only sent from the analysis service."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.username = username
        self.password = password
        self._owns_client = http_client is None
        self.client = http_client or httpx.Client(
            timeout=httpx.Timeout(
                get_settings().navidrome_timeout_seconds,
                read=get_settings().navidrome_read_timeout_seconds,
            ),
            follow_redirects=True,
        )

    def _auth(self) -> dict[str, str]:
        salt = secrets.token_hex(8)
        token = hashlib.md5((self.password + salt).encode(), usedforsecurity=False).hexdigest()
        return {
            "u": self.username,
            "t": token,
            "s": salt,
            "v": "1.16.1",
            "c": "echora",
            "f": "json",
        }

    def _endpoint(self, method: str) -> str:
        return urljoin(self.base_url, f"rest/{method}.view")

    def _request(self, method: str, **params: object) -> dict[str, object]:
        state = _batch_state.get()
        if state:
            state["check"]()
        response = self.client.get(self._endpoint(method), params={**self._auth(), **params})
        response.raise_for_status()
        payload = response.json()["subsonic-response"]
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"Navidrome error: {payload.get('error', {}).get('message', 'unknown')}"
            )
        return payload

    @staticmethod
    def _track(song: dict[str, object]) -> NavidromeTrack:
        return NavidromeTrack(
            id=str(song["id"]),
            title=str(song.get("title") or "Unknown title"),
            artist=song.get("artist"),
            album=song.get("album"),
            duration=float(song.get("duration") or 0),
            year=int(song["year"]) if song.get("year") else None,
            genre=song.get("genre"),
            suffix=song.get("suffix"),
            path=song.get("path"),
            raw=song,
        )

    def ping(self) -> str:
        payload = self._request("ping")
        return str(payload.get("serverVersion") or payload.get("version") or "Subsonic")

    def random_tracks(self, size: int = 100) -> list[NavidromeTrack]:
        payload = self._request("getRandomSongs", size=size)
        songs = payload.get("randomSongs", {}).get("song", [])
        return [self._track(song) for song in songs]

    def all_tracks(self, page_size: int = 500, maximum: int | None = None) -> list[NavidromeTrack]:
        if page_size <= 0 or (maximum is not None and maximum <= 0):
            raise ValueError("Pagination limits must be positive")
        tracks: list[NavidromeTrack] = []
        seen: set[str] = set()
        while True:
            payload = self._request(
                "search3",
                query="",
                songCount=page_size,
                songOffset=len(tracks),
                artistCount=0,
                albumCount=0,
            )
            result = payload.get("searchResult3")
            if not isinstance(result, dict):
                raise RuntimeError("Incomplete Navidrome catalog response")
            songs = result.get("song", [])
            if not isinstance(songs, list) or len(songs) > page_size:
                raise RuntimeError("Invalid Navidrome catalog page")
            if not songs:
                return tracks
            page = [self._track(song) for song in songs]
            ids = {song.id for song in page}
            if len(ids) != len(page) or ids & seen:
                raise RuntimeError("Repeating Navidrome catalog pagination")
            seen.update(ids)
            tracks.extend(page)
            if maximum is not None and len(tracks) > maximum:
                raise RuntimeError("Navidrome catalog exceeds safety limit")
            # Probe beyond short pages too: servers may silently cap page size.

    def tracks(self, song_ids: list[str]) -> list[NavidromeTrack]:
        tracks = []
        for song_id in song_ids:
            payload = self._request("getSong", id=song_id)
            tracks.append(self._track(payload["song"]))
        return tracks

    def lyrics(self, song_id: str) -> dict[str, object]:
        try:
            # Navidrome omits translation layers and syllable cues unless opted in.
            payload = self._request("getLyricsBySongId", id=song_id, enhanced=True)
            entries = payload.get("lyricsList", {}).get("structuredLyrics", [])
            if entries:
                entry = next(
                    (item for item in entries if item.get("kind", "main") in ("main", "")), None
                )
                if entry is None:
                    return {
                        "status": "missing",
                        "text": None,
                        "language": None,
                        "synced": False,
                        "lines": [],
                        "source": "getLyricsBySongId",
                        "structured_lyrics": entries,
                    }
                lines = entry.get("line", [])
                text = "\n".join(str(line.get("value") or "") for line in lines).strip()
                return {
                    "status": "available" if text else "missing",
                    "text": text or None,
                    "language": entry.get("lang"),
                    "synced": bool(entry.get("synced")),
                    "lines": [
                        {"start_ms": line.get("start"), "text": line.get("value") or ""}
                        for line in lines
                    ],
                    "source": "getLyricsBySongId",
                    "structured_lyrics": entries,
                }
        except Exception:
            pass
        try:
            payload = self._request("getLyrics", songId=song_id)
            entry = payload.get("lyrics") or {}
            text = str(entry.get("value") or "").strip()
            return {
                "status": "available" if text else "missing",
                "text": text or None,
                "language": None,
                "synced": False,
                "lines": [],
                "source": "getLyrics",
            }
        except Exception:
            return {
                "status": "unavailable",
                "text": None,
                "language": None,
                "synced": False,
                "lines": [],
                "source": "navidrome",
            }

    def playlists(self) -> list[dict[str, object]]:
        payload = self._request("getPlaylists")
        return list(payload.get("playlists", {}).get("playlist", []))

    def replace_playlist(
        self, name: str, song_ids: list[str], playlist_id: str | None = None
    ) -> str:
        # Subsonic createPlaylist updates in place when playlistId is supplied.
        # Creating by name first and deleting the old ID leaves duplicate
        # playlists when deletion fails or clients refresh between requests.
        params: dict[str, object] = {"songId": song_ids}
        if playlist_id:
            params["playlistId"] = playlist_id
        else:
            params["name"] = name
        payload = self._request("createPlaylist", **params)
        playlist = payload.get("playlist") or {}
        returned_id = str(playlist.get("id") or playlist_id or "")
        if not returned_id:
            raise RuntimeError("Navidrome did not return the playlist ID")
        return returned_id

    def delete_playlist(self, playlist_id: str) -> None:
        self._request("deletePlaylist", id=playlist_id)

    def catalog(self) -> dict[str, list[dict[str, object]]]:

        album_payload = self._request("getAlbumList2", type="alphabeticalByName", size=500)
        artist_payload = self._request("getArtists")
        albums = album_payload.get("albumList2", {}).get("album", [])
        indexes = artist_payload.get("artists", {}).get("index", [])
        artists = [artist for index in indexes for artist in index.get("artist", [])]
        return {"albums": albums, "artists": artists}

    def cover_art(self, cover_id: str, size: int = 160) -> tuple[bytes, str]:
        response = self.client.get(
            self._endpoint("getCoverArt"),
            params={**self._auth(), "id": cover_id, "size": size},
        )
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "image/jpeg")

    def scrobble(self, song_id: str, submission: bool = True) -> None:
        """Subsonic scrobble: submission=False marks the song as now playing,
        submission=True records a completed play (feeds Navidrome play counts)."""
        self._request("scrobble", id=song_id, submission="true" if submission else "false")

    def transcode_chunks(
        self, song_id: str, max_bit_rate: int = 320, range_header: str | None = None
    ) -> Iterator[bytes]:
        headers = {"Range": range_header} if range_header else {}
        with self.client.stream(
            "GET",
            self._endpoint("stream"),
            params={
                **self._auth(),
                "id": song_id,
                "format": "mp3",
                "maxBitRate": max_bit_rate,
                "estimateContentLength": "true",
            },
            headers=headers,
        ) as response:
            response.raise_for_status()
            yield from response.iter_bytes(64 * 1024)

    def open_transcode_stream(
        self, song_id: str, max_bit_rate: int | None = 320, range_header: str | None = None
    ) -> httpx.Response:
        stream_params = {
            **self._auth(),
            "id": song_id,
            "format": "raw" if max_bit_rate is None else "mp3",
            "estimateContentLength": "true",
        }
        if max_bit_rate is not None:
            stream_params["maxBitRate"] = max_bit_rate
        request = self.client.build_request(
            "GET",
            self._endpoint("stream"),
            params=stream_params,
            headers={"Range": range_header} if range_header else {},
        )
        response = self.client.send(request, stream=True)
        response.raise_for_status()
        if "application/json" in response.headers.get("content-type", ""):
            try:
                response.read()
                payload = response.json().get("subsonic-response", {})
                message = payload.get("error", {}).get(
                    "message", "Navidrome returned JSON instead of audio"
                )
            except Exception:
                message = "Navidrome returned JSON instead of audio"
            response.close()
            raise RuntimeError(str(message))
        return response

    def transcode_range(
        self, song_id: str, range_header: str, max_bit_rate: int = 320
    ) -> tuple[bytes, dict[str, str], int]:
        response = self.client.get(
            self._endpoint("stream"),
            params={
                **self._auth(),
                "id": song_id,
                "format": "mp3",
                "maxBitRate": max_bit_rate,
                "estimateContentLength": "true",
            },
            headers={"Range": range_header},
        )
        response.raise_for_status()
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"content-length", "content-range", "accept-ranges"}
        }
        return response.content, headers, response.status_code

    def transcode(self, song_id: str, max_bit_rate: int | None = 320) -> tuple[bytes, str]:
        stream_params = {
            **self._auth(),
            "id": song_id,
            "format": "raw" if max_bit_rate is None else "mp3",
            "estimateContentLength": "true",
        }
        if max_bit_rate is not None:
            stream_params["maxBitRate"] = max_bit_rate
        response = self.client.get(
            self._endpoint("stream"),
            params=stream_params,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "audio/mpeg")
        if "application/json" in content_type:
            try:
                payload = response.json().get("subsonic-response", {})
                message = payload.get("error", {}).get(
                    "message", "Navidrome returned JSON instead of audio"
                )
            except Exception:
                message = "Navidrome returned JSON instead of audio"
            raise RuntimeError(str(message))
        return response.content, content_type

    def audio_bytes(self, song_id: str) -> bytes:
        state = _batch_state.get()
        path = None
        if state:
            state["check"]()
            key = hashlib.sha256(repr((self.base_url, self.username, song_id)).encode()).hexdigest()
            path = state["directory"] / key
            if path.exists():
                return path.read_bytes()
        response = self.client.get(
            self._endpoint("stream"),
            params={
                **self._auth(),
                "id": song_id,
                "format": "raw",
                "estimateContentLength": "true",
            },
        )
        response.raise_for_status()
        audio = response.content
        if state:
            state["check"]()
            if state["size"] + len(audio) <= state["limit"]:
                path.write_bytes(audio)
                state["size"] += len(audio)
        return audio

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "NavidromeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def media_navidrome_client(base_url: str, username: str, password: str) -> NavidromeClient:
    """Return a Navidrome client backed by the process-wide media connection pool."""
    return NavidromeClient(base_url, username, password, http_client=_media_http_client)
