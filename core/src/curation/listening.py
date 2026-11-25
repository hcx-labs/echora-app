"""Listening history.

Last.fm listens and top tracks, Navidrome play counts, and matching them to the library.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx

from core.integrations.config import LastfmConfig


@dataclass(frozen=True)
class Listen:
    """One scrobbled play from Last.fm."""

    artist: str
    title: str
    played_at: datetime


_cache: dict[tuple[str, str, int], tuple[float, list[Listen]]] = {}
_cache_lock = threading.Lock()


def _normalize(value: str | None) -> str:
    folded = (
        unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode().casefold()
    )
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def recent_listens(username: str, api_key: str, since: datetime) -> list[Listen]:
    since_unix = int(since.timestamp())
    cache_key = (
        username.casefold(),
        hashlib.sha256(api_key.encode()).hexdigest(),
        since_unix // 3600,
    )
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 300:
            return list(cached[1])
    listens: list[Listen] = []
    page = 1
    with httpx.Client(timeout=LastfmConfig.load().history_timeout_seconds) as client:
        while page <= 20:
            response = client.get(
                "https://ws.audioscrobbler.com/2.0/",
                params={
                    "method": "user.getrecenttracks",
                    "user": username,
                    "api_key": api_key,
                    "format": "json",
                    "from": since_unix,
                    "limit": 200,
                    "page": page,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(
                    str(payload.get("message") or "Last.fm rejected the history request")
                )
            recent = payload.get("recenttracks", {})
            tracks = recent.get("track", [])
            for track in tracks:
                timestamp = track.get("date", {}).get("uts")
                if not timestamp:
                    continue
                listens.append(
                    Listen(
                        artist=str(track.get("artist", {}).get("#text") or ""),
                        title=str(track.get("name") or ""),
                        played_at=datetime.fromtimestamp(int(timestamp), tz=UTC),
                    )
                )
            pages = int(recent.get("@attr", {}).get("totalPages") or 1)
            if page >= pages:
                break
            page += 1
    with _cache_lock:
        _cache[cache_key] = (time.monotonic(), list(listens))
    return listens


def track_listen_counts(
    rows: list[dict[str, object]],
    listens: list[Listen],
    timezone_name: str,
    period_start: str | None = None,
    period_end: str | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    exact: dict[tuple[str, str], str] = {}
    title_ids: dict[str, list[str]] = {}
    for row in rows:
        identifier = str(row["id"])
        title = _normalize(str(row.get("title") or ""))
        exact[(_normalize(str(row.get("artist") or "")), title)] = identifier
        title_ids.setdefault(title, []).append(identifier)
    all_counts: Counter[str] = Counter()
    period_counts: Counter[str] = Counter()
    zone = ZoneInfo(timezone_name)
    start_minutes = _minutes(period_start) if period_start else None
    end_minutes = _minutes(period_end) if period_end else None
    for listen in listens:
        title = _normalize(listen.title)
        identifier = exact.get((_normalize(listen.artist), title))
        if identifier is None and len(title_ids.get(title, [])) == 1:
            identifier = title_ids[title][0]
        if identifier is None:
            continue
        all_counts[identifier] += 1
        if start_minutes is not None and end_minutes is not None:
            local = listen.played_at.astimezone(zone)
            minute = local.hour * 60 + local.minute
            in_period = (
                start_minutes <= minute < end_minutes
                if start_minutes < end_minutes
                else minute >= start_minutes or minute < end_minutes
            )
            if in_period:
                period_counts[identifier] += 1
    return dict(all_counts), dict(period_counts)


@dataclass(frozen=True)
class TopTrack:
    """A track of a Last.fm top-tracks chart, with its play count for the period."""

    artist: str
    title: str
    play_count: int


TOP_TRACK_PERIODS = {"7day", "1month", "3month", "6month", "12month", "overall"}
_top_cache: dict[tuple[str, str, str, int], tuple[float, list[TopTrack]]] = {}


def top_tracks(username: str, api_key: str, period: str, limit: int) -> list[TopTrack]:
    """Last.fm's own ranking of the user's most played tracks for a period."""
    if period not in TOP_TRACK_PERIODS:
        raise ValueError("Unknown Last.fm period")
    cache_key = (username.casefold(), hashlib.sha256(api_key.encode()).hexdigest(), period, limit)
    with _cache_lock:
        cached = _top_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 600:
            return list(cached[1])
    with httpx.Client(timeout=LastfmConfig.load().history_timeout_seconds) as client:
        response = client.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "user.gettoptracks",
                "user": username,
                "api_key": api_key,
                "format": "json",
                "period": period,
                "limit": limit,
            },
        )
        response.raise_for_status()
        payload = response.json()
    if payload.get("error"):
        raise RuntimeError(str(payload.get("message") or "Last.fm rejected the top tracks request"))
    entries = [
        TopTrack(
            artist=str((track.get("artist") or {}).get("name") or ""),
            title=str(track.get("name") or ""),
            play_count=int(track.get("playcount") or 0),
        )
        for track in payload.get("toptracks", {}).get("track", [])
    ]
    with _cache_lock:
        _top_cache[cache_key] = (time.monotonic(), list(entries))
    return entries


def navidrome_play_counts(songs: list[dict[str, object]]) -> list[tuple[str, int]]:
    """Rank Navidrome songs by their all-time ``playCount``, most played first.

    Navidrome keeps only a running total per song, so this has no time periods.
    Unplayed songs are dropped; ties fall back to the most recent ``played``
    time, then the song id so the order is stable.
    """
    ranked: list[tuple[int, str, str]] = []
    for song in songs:
        try:
            count = int(song.get("playCount") or 0)
        except (TypeError, ValueError):
            continue
        if count > 0 and song.get("id"):
            ranked.append((count, str(song.get("played") or ""), str(song["id"])))
    # Stable sorts, least significant key first: id, then newest play, then count.
    ranked.sort(key=lambda item: item[2])
    ranked.sort(key=lambda item: item[1], reverse=True)
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(song_id, count) for count, _, song_id in ranked]


def match_navidrome_play_counts(
    rows: list[dict[str, object]], ranked: list[tuple[str, int]]
) -> list[tuple[dict[str, object], int]]:
    """Map ranked Navidrome song ids onto library rows by their source id."""
    by_source = {str(row["source_id"]): row for row in rows if row.get("source_id")}
    matched: list[tuple[dict[str, object], int]] = []
    seen: set[str] = set()
    for song_id, count in ranked:
        row = by_source.get(song_id)
        if row is None or str(row["id"]) in seen:
            continue
        seen.add(str(row["id"]))
        matched.append((row, count))
    return matched


def match_top_tracks(
    rows: list[dict[str, object]], entries: list[TopTrack]
) -> list[tuple[dict[str, object], int]]:
    """Map ranked Last.fm entries onto library rows, keeping Last.fm's order.

    Matching mirrors track_listen_counts: exact artist and title first, then a
    title that is unique in the library. Each library track appears once.
    """
    exact: dict[tuple[str, str], dict[str, object]] = {}
    by_title: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        title = _normalize(str(row.get("title") or ""))
        exact.setdefault((_normalize(str(row.get("artist") or "")), title), row)
        by_title.setdefault(title, []).append(row)
    matched: list[tuple[dict[str, object], int]] = []
    seen: set[str] = set()
    for entry in entries:
        title = _normalize(entry.title)
        row = exact.get((_normalize(entry.artist), title))
        if row is None and len(by_title.get(title, [])) == 1:
            row = by_title[title][0]
        if row is None or str(row["id"]) in seen:
            continue
        seen.add(str(row["id"]))
        matched.append((row, entry.play_count))
    return matched


def _minutes(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)
