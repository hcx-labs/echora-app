from backend.library.track_info import track_info as _track_info


def test_track_info_collects_tags_analysis_lyrics_and_themes():
    source = {
        "source_data": {
            "year": 2012,
            "genres": [{"name": "K-Pop"}],
            "moods": [],
            "contributors": [
                {"role": "composer", "artist": {"name": "G-DRAGON"}},
                {"role": "composer", "artist": {"name": "G-DRAGON"}},
                {"role": "lyricist", "artist": {"name": "G-DRAGON"}},
            ],
            "isrc": ["KRA341204052"],
            "suffix": "FLAC",
            "bitDepth": 24,
            "samplingRate": 48000,
            "channelCount": 2,
            "albumArtists": [{"name": "BIGBANG"}],
            "explicitStatus": "",
        }
    }
    descriptors = {
        "descriptors": {
            "rhythm": {"bpm": 142.0},
            "key": {"tonic": "C", "scale": "major"},
            "loudness": {"integrated_lufs": -9.0, "range_lu": 7.6},
        }
    }
    lyrics = {
        "text": "hello",
        "language": "ko",
        "provenance": {"synced": True},
        "karaoke_lines": True,
    }
    info = _track_info(
        source, descriptors, lyrics, {"description": "About love.", "themes": ["love"]}
    )
    assert info["tags"]["genres"] == ["K-Pop"]
    assert info["tags"]["credits"] == {"composer": ["G-DRAGON"], "lyricist": ["G-DRAGON"]}
    assert info["tags"]["isrc"] == "KRA341204052"
    assert info["tags"]["album_artists"] == ["BIGBANG"]
    assert info["audio"]["lossless"] is True and info["audio"]["codec"] == "flac"
    assert info["analysis"]["key"] == "C major" and info["analysis"]["bpm"] == 142.0
    assert info["lyrics"] == {
        "available": True,
        "language": "ko",
        "source": None,
        "synced": True,
        "karaoke": True,
    }
    assert info["themes"] == {"description": "About love.", "tags": ["love"]}


def test_track_info_without_analysis_or_lyrics():
    info = _track_info({"source_data": {}}, None, None, None)
    assert info["analysis"]["key"] is None
    assert info["lyrics"]["available"] is False
    assert info["themes"] is None
