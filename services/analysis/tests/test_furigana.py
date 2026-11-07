from echora_analysis.furigana import romaji_to_hiragana, ruby_segments
from echora_analysis.karaoke_pipeline import build_lines_from_alignment_document


def _morae(*units):
    return [
        {"unit": unit, "start_ms": 100 * i, "end_ms": 100 * i + 100} for i, unit in enumerate(units)
    ]


def _reading(surface, *units):
    return [
        (s["start"], s["end"], s["text"], [m["text"] for m in s["morae"]])
        for s in ruby_segments(surface, _morae(*units))
    ]


def test_romaji_units_to_hiragana():
    assert romaji_to_hiragana("jo") == "じょ"
    assert romaji_to_hiragana("nat") == "なっ"
    assert romaji_to_hiragana("itta") == "いった"
    assert romaji_to_hiragana("matcha") == "まっちゃ"
    assert romaji_to_hiragana("kyo") == "きょ"
    assert romaji_to_hiragana("n") == "ん"
    assert romaji_to_hiragana("x") is None


def test_kanji_groups_get_their_timed_reading():
    assert _reading("彼女", "ka", "no", "jo") == [(0, 2, "かのじょ", ["か", "の", "じょ"])]
    assert _reading("東京", "to", "u", "kyo", "u") == [
        (0, 2, "とうきょう", ["と", "う", "きょ", "う"])
    ]


def test_kana_inside_a_group_anchor_the_reading():
    assert _reading("飛び交", "to", "bi", "ka") == [(0, 1, "と", ["と"]), (2, 3, "か", ["か"])]
    # The sokuon of the last mora belongs to the trailing っ, not the kanji.
    assert _reading("思っ", "o", "mot") == [(0, 1, "おも", ["お", "も"])]


def test_no_ruby_for_kana_or_unplaceable_readings():
    assert _reading("いったい", "itta", "i") == []
    assert _reading("彼女", "ka", "no", "x") == []
    # Kana that do not match the reading: one reading over the whole token.
    assert _reading("飛び", "to", "ku") == [(0, 2, "とく", ["と", "く"])]


def _document(language, text, tokens):
    return {
        "diagnostics": {"normalization_language": language},
        "alignment": {"lines": [{"text": text, "start_ms": 0, "end_ms": 900, "tokens": tokens}]},
    }


def test_lines_carry_ruby_for_japanese_only():
    tokens = [
        {"text": "彼女", "start_ms": 0, "end_ms": 100, "acoustic_token": "ka"},
        {"text": "", "start_ms": 100, "end_ms": 200, "acoustic_token": "no"},
        {"text": "", "start_ms": 200, "end_ms": 300, "acoustic_token": "jo"},
        {"text": "は", "start_ms": 300, "end_ms": 400, "acoustic_token": "wa"},
    ]
    line = build_lines_from_alignment_document(_document("ja", "「彼女は」", tokens))[0]
    first = line["syllables"][0]
    assert first["text"] == "「彼女"
    assert first["ruby"] == [
        {
            "start": 1,
            "end": 3,
            "text": "かのじょ",
            "morae": [
                {"text": "か", "start_ms": 0, "end_ms": 100},
                {"text": "の", "start_ms": 100, "end_ms": 200},
                {"text": "じょ", "start_ms": 200, "end_ms": 300},
            ],
        }
    ]
    assert "ruby" not in line["syllables"][1]
    chinese = build_lines_from_alignment_document(_document("zh", "彼女は", tokens))[0]
    assert all("ruby" not in syllable for syllable in chinese["syllables"])
