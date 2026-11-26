"""Backing vocals in the plugin's TTML export (core.lyrics.backing splits them)."""

from xml.etree import ElementTree as ET

import backend.plugin.lyrics as plugin_lyrics


def test_untimed_export_puts_backing_in_an_x_bg_span_without_brackets():
    row = {
        "text": "Hello (hey) I am (hey)",
        "provenance": {"lines": [{"text": "Hello (hey) I am (hey)", "start_ms": 1000}]},
        "karaoke_lines": [],
        "language": "en",
    }
    root = ET.fromstring(plugin_lyrics.ttml(row, []))
    (paragraph,) = root.iter(f"{{{plugin_lyrics.TT}}}p")
    assert paragraph.text == "Hello I am "
    (span,) = list(paragraph)
    assert span.get(f"{{{plugin_lyrics.TTM}}}role") == "x-bg" and span.text == "hey, hey"


def test_ttml_applies_the_profiles_karaoke_smoothing():
    from backend.settings.plugin_integration import RankingProfile

    row = {
        "text": "hello",
        "karaoke_lines": [
            {
                "text": "hello",
                "start_ms": 0,
                "end_ms": 500,
                "syllables": [
                    {"text": "hel", "start_ms": 0, "end_ms": 100},
                    {"text": "lo", "start_ms": 300, "end_ms": 500},
                ],
            }
        ],
        "language": "en",
    }

    def begins_ends(profile):
        (item,) = plugin_lyrics.lyrics_response(row, [], profile)["lyrics"]
        spans = ET.fromstring(item["text"]).iter(f"{{{plugin_lyrics.TT}}}span")
        return [(s.get("begin"), s.get("end")) for s in spans]

    assert begins_ends(RankingProfile())[0] == ("00:00:00.000", "00:00:00.100")
    assert begins_ends(RankingProfile(karaoke_close_gaps=True))[0] == (
        "00:00:00.000",
        "00:00:00.300",
    )
    assert row["karaoke_lines"][0]["syllables"][0]["end_ms"] == 100
