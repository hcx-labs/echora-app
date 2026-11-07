"""No model/checkpoint loading: exercise the shared vendor normalizer."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "fa_kara"))
import haruraw2norm as norm


def test_janome_word_readings_and_particles():
    items = norm.process_haruhi_line("宝石に憧れ私は東京へ行く", "ja")
    words = {i["orig"]: i.get("pron") for i in items}
    assert "".join(i.get("pron", "") for i in norm.process_haruhi_line("宝石", "ja")) == "houseki"
    assert "".join(i.get("pron", "") for i in norm.process_haruhi_line("憧れ", "ja")) == "akogare"
    assert words["宝石"] == "ho"
    assert words["憧"] == "a"
    assert words["は"] == "wa"
    assert words["へ"] == "e"
    assert "".join(i["orig"] for i in items) == "宝石に憧れ私は東京へ行く"
    assert items == norm.process_haruhi_line("宝石に憧れ私は東京へ行く", "auto")


def test_chinese_context_before_character_mapping():
    import pypinyin

    with patch.object(pypinyin, "pinyin", wraps=pypinyin.pinyin) as contextual:
        items = norm.process_haruhi_line("银行 重庆", "auto")
    assert [call.args[0] for call in contextual.call_args_list] == ["银行", "重庆"]
    assert [(i["orig"], i.get("pron")) for i in items if i.get("pron")] == [
        ("银", "in"),
        ("行", "hang"),
        ("重", "chong"),
        ("庆", "ching"),
    ]
    # Traditional lyrics get the same phrase-level reading (OpenCC t2s before pypinyin).
    assert [i.get("pron") for i in norm.process_haruhi_line("銀行", "auto")] == ["in", "hang"]


def test_mandarin_spelling_is_tone_less_and_by_sound():
    prons = [i["pron"] for i in norm.process_haruhi_line("来了知道去学习绿女", "zh")]
    assert prons == ["lai", "le", "jr", "dao", "chyu", "shyue", "shi", "lyu", "nyu"]


def test_cantonese_by_hint_or_marker():
    assert [i["pron"] for i in norm.process_haruhi_line("一個", "yue")] == ["yat", "go"]
    assert [i["pron"] for i in norm.process_haruhi_line("一個", "auto")] == ["i", "ge"]
    assert [i["pron"] for i in norm.process_haruhi_line("唔係一個", "auto")] == [
        "m",
        "hai",
        "yat",
        "go",
    ]


def test_kana_anywhere_in_line_makes_han_runs_japanese():
    assert (
        "".join(i["pron"] for i in norm.process_haruhi_line("東京 へ", "auto") if i.get("pron"))
        == "toukyoue"
    )
    assert [i["pron"] for i in norm.process_haruhi_line("東京 へ", "zh") if i.get("pron")][
        0
    ] == "dong"


def test_thai_syllables_and_repeat_mark():
    items = [i for i in norm.process_haruhi_line("ความรัก ดีๆ", "auto") if i.get("pron")]
    assert [(i["orig"], i["pron"]) for i in items] == [
        ("ความ", "khwam"),
        ("รัก", "rak"),
        ("ดี", "di"),
        ("ๆ", "di"),
    ]


def test_arabic_script_long_vowels_and_no_apostrophes():
    items = [i for i in norm.process_haruhi_line("حبيبي عمر", "auto") if i.get("pron")]
    assert [i["pron"] for i in items] == ["hbibi", "mr"]


def test_other_scripts_use_uroman():
    prons = [
        i["pron"] for i in norm.process_haruhi_line("Я сказал щёки நெஞ்சம்", "auto") if i.get("pron")
    ]
    assert prons == ["ya", "skazal", "shchyoki", "nenycam"]


def test_turkish_letters_select_turkish_latin_rules():
    assert [i["pron"] for i in norm.process_haruhi_line("hiç ışık", "auto") if i.get("pron")] == [
        "hich",
        "ishik",
    ]
    assert [i["pron"] for i in norm.process_haruhi_line("garçon", "auto") if i.get("pron")] == [
        "garson"
    ]


def test_annotations_remain_overrides_in_mixed_scripts():
    for lang in ("auto", "ja", "ko", "es"):
        items = norm.process_haruhi_line("한{宝石|アコガレ}[重庆|hello]", lang)
        assert "".join(i.get("pron", "") for i in items) == "hanakogarehello"
        assert "".join(i["orig"] for i in items) == "한宝石重庆"


def test_ruby_sokuon_across_annotation_boundaries():
    for text in ("{勝|かっ}{手|て}", "{勝|かっ}手"):
        items = norm.process_haruhi_line(text, "ja")
        assert "".join(i.get("pron", "") for i in items) == "katte"


def test_code_switch_and_unknown_surfaces():
    text = "한宝石に憧れ银行重庆 café नमस्ते مرحبا Ж"
    items = norm.process_haruhi_line(text, "auto")
    assert "".join(i["orig"] for i in items) == text
    index = next(n for n, i in enumerate(items) if i["orig"] == "宝石")
    assert "".join(i["pron"] for i in items[index : index + 4]) == "houseki"
    for surface in ("한", "café", "नमस्ते", "مرحبا", "Ж"):
        assert next(i for i in items if i["orig"] == surface)["pron"]


def test_latin_fallback_keeps_accents_and_avoids_cmu():
    for lang in ("auto", "es", "fr", "ko"):
        with patch.object(
            norm, "process_english_word", side_effect=AssertionError("CMU forbidden")
        ):
            items = norm.process_haruhi_line("no café canción cafe\u0301", lang)
        assert "".join(i["orig"] for i in items) == "no café canción cafe\u0301"
        assert [i["pron"] for i in items if i.get("pron")] == ["no", "cafe", "cancion", "cafe"]
    assert norm.process_haruhi_line("no", "en")[0]["pron"] == "no"


def test_han_only_policy_is_run_local():
    chinese = norm.process_haruhi_line("银行", "auto")
    assert norm.process_haruhi_line("한银行", "auto")[1:] == chinese
    assert [i["pron"] for i in norm.process_haruhi_line("宝石", "ja")] == ["ho", "u", "se", "ki"]
