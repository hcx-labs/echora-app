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
