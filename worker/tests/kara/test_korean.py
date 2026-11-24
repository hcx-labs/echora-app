import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[2] / "src" / "kara" / "vendor" / "fa_kara"
sys.path.insert(0, str(VENDOR))

from haruraw2norm import process_haruhi_line  # noqa: E402


def test_japanese_auto_detection_pronounces_kanji_and_kana():
    result = process_haruhi_line("世界中全部 変えてしまえば\n", "auto")
    spoken = [item for item in result if item.get("pron")]

    assert "".join(item["orig"] for item in spoken).replace(" ", "") == "世界中全部変えてしまえば"
    assert all(item["pron"] for item in spoken)


def test_korean_auto_detection_preserves_hangul_and_adds_latin_pronunciation():
    result = process_haruhi_line("사랑해\n", "auto")
    spoken = [item for item in result if item.get("pron")]

    assert "".join(item["orig"] for item in spoken) == "사랑해"
    assert [item["pron"] for item in spoken] == ["sa", "rang", "hae"]


def test_urdu_auto_detection_preserves_words_and_adds_latin_pronunciation():
    result = process_haruhi_line("کالی کالی رات میں\n", "auto")
    spoken = [item for item in result if item.get("pron")]

    assert [item["orig"] for item in spoken] == ["کالی", "کالی", "رات", "میں"]
    assert [item["pron"] for item in spoken] == ["kali", "kali", "rat", "min"]


def test_hindi_auto_detection_preserves_words_and_adds_latin_pronunciation():
    result = process_haruhi_line("मैं रंग शरबतों का\n", "auto")
    spoken = [item for item in result if item.get("pron")]

    assert [item["orig"] for item in spoken] == ["मैं", "रंग", "शरबतों", "का"]
    assert [item["pron"] for item in spoken] == ["maim", "ramg", "sharabatom", "kaa"]


def test_mixed_devanagari_gurmukhi_line_keeps_both_scripts():
    result = process_haruhi_line("ਹੋ ਯਾਰਾ तुझे प्यार\n", "auto")
    spoken = [item for item in result if item.get("pron")]

    assert [item["orig"] for item in spoken] == ["ਹੋ", "ਯਾਰਾ", "तुझे", "प्यार"]
    assert all(item["pron"].isascii() for item in spoken)


def test_korean_mixed_english_keeps_both_display_scripts():
    result = process_haruhi_line("너를 love해\n", "auto")
    spoken = [item for item in result if item.get("pron")]

    assert "".join(item["orig"] for item in spoken) == "너를love해"
    assert all(item["pron"].isascii() for item in spoken)


def test_inference_labels_hangul_by_pronunciation_like_training():
    import main as vendor

    records, _ = vendor.normalize_source_lines(["같이 먹어요"], "auto", 0, 1)
    assert [item["pron"] for item in records if item.get("pron")] == [
        "ga",
        "chi",
        "meo",
        "geo",
        "yo",
    ]
