from xml.etree import ElementTree as ET

import echora_analysis.plugin_lyrics as plugin_lyrics
import pytest

from core.lyrics.backing import separate_text, separate_voices, tag_bracketed_backing


def split(texts, voices=None):
    syllables = [
        {"text": t, "start_ms": i * 100, "end_ms": i * 100 + 90} for i, t in enumerate(texts)
    ]
    if voices:
        for syllable, voice in zip(syllables, voices):
            if voice:
                syllable["voice"] = voice
    else:
        tag_bracketed_backing([{"text": "".join(texts), "syllables": syllables}])
    lead, backing = separate_voices(syllables)
    return "".join(s["text"] for s in lead), "".join(s["text"] for s in backing), backing


def test_two_bracketed_parts_become_comma_separated_backing():
    lead, backing, syllables = split(["Hello ", "(hey) ", "I ", "am ", "(hey)"])
    assert (lead, backing) == ("Hello I am", "hey, hey")
    assert [s["start_ms"] for s in syllables] == [100, 400]


@pytest.mark.parametrize(
    ("texts", "voices", "expected"),
    [
        # Brackets glued to lead words, as the aligner stores them.
        (
            ["Wait (", "don't ", "let ", "go), ", "and ", "pre", "tend (", "don't ", "let ", "go)"],
            None,
            ("Wait and pretend", "don't let go, don't let go"),
        ),
        # Navidrome background cues after the lead words, each with its own brackets.
        (
            ["Hello ", "I ", "am ", "(hey) ", "(hey)"],
            [None, None, None, 2, 2],
            ("Hello I am", "hey, hey"),
        ),
        (["(One, ", "two, ", "three)"], None, ("", "One, two, three")),
        (["No ", "backing"], None, ("No backing", "")),
    ],
)
def test_split_cases(texts, voices, expected):
    assert split(texts, voices)[:2] == expected


def test_text_split():
    assert separate_text("Hello (hey) I am (hey)") == ("Hello I am", "hey, hey")
    assert separate_text("Is he far away (is he far away?), is he okay? (Is he okay?)") == (
        "Is he far away, is he okay?",
        "is he far away? Is he okay?",
    )
    assert separate_text("plain") == ("plain", "")


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
