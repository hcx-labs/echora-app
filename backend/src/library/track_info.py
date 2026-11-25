"""The song-info panel's facts, from Navidrome's tags, audio analysis, lyrics and themes."""

from __future__ import annotations

LOSSLESS_CODECS = {"flac", "alac", "wav", "aiff", "ape", "wv"}


def track_info(source: dict, descriptors: dict | None, lyrics: dict | None, themes: dict | None):
    """The song-info panel's facts, from Navidrome's tags, audio analysis, lyrics and themes."""
    data = source.get("source_data") or {}

    def names(value):
        return [
            str(item.get("name") if isinstance(item, dict) else item)
            for item in (value or [])
            if (item.get("name") if isinstance(item, dict) else item)
        ]

    credits: dict[str, list[str]] = {}
    for contributor in data.get("contributors") or []:
        role = str(contributor.get("role") or "").strip()
        name = str((contributor.get("artist") or {}).get("name") or "").strip()
        if role and name and name not in credits.setdefault(role, []):
            credits[role].append(name)
    codec = str(data.get("suffix") or "").lower()
    found = (descriptors or {}).get("descriptors") or {}
    key = found.get("key") or {}
    loudness = found.get("loudness") or {}
    provenance = (lyrics or {}).get("provenance") or {}
    return {
        "tags": {
            "year": data.get("year") or None,
            "genres": names(data.get("genres")) or ([data["genre"]] if data.get("genre") else []),
            "moods": names(data.get("moods")),
            "track": data.get("track") or None,
            "disc": data.get("discNumber") or None,
            "album_artists": names(data.get("albumArtists")),
            "credits": credits,
            "isrc": (data.get("isrc") or [None])[0]
            if isinstance(data.get("isrc"), list)
            else data.get("isrc"),
            "explicit": data.get("explicitStatus") == "explicit",
            "comment": data.get("comment") or None,
            "play_count": data.get("playCount"),
            "last_played": data.get("played"),
            "added": data.get("created"),
            "size_bytes": data.get("size"),
            "musicbrainz_id": data.get("musicBrainzId") or None,
        },
        "audio": {
            "codec": codec or None,
            "lossless": codec in LOSSLESS_CODECS,
            "bit_rate_kbps": data.get("bitRate"),
            "bit_depth": data.get("bitDepth"),
            "sample_rate_hz": data.get("samplingRate"),
            "channels": data.get("channelCount"),
            "replay_gain": data.get("replayGain") or None,
        },
        "analysis": {
            "bpm": (found.get("rhythm") or {}).get("bpm"),
            "key": f"{key['tonic']} {key.get('scale') or ''}".strip() if key.get("tonic") else None,
            "integrated_lufs": loudness.get("integrated_lufs"),
            "loudness_range_lu": loudness.get("range_lu"),
            "true_peak_dbtp": found.get("true_peak_dbtp"),
        },
        "lyrics": {
            "available": bool(lyrics and str(lyrics.get("text") or "").strip()),
            "language": (lyrics or {}).get("language"),
            "source": "ai" if provenance.get("ai_generated") else provenance.get("provider"),
            "synced": bool(provenance.get("synced")),
            "karaoke": bool((lyrics or {}).get("karaoke_lines")),
        },
        "themes": {"description": themes["description"], "tags": themes["themes"]}
        if themes
        else None,
    }
