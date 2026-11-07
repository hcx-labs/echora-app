"""Label rules v2 of the shared vendor normalizer (data/finetune-curation/DATASET_FIXES_2026-10-18.md).

v1 stays the default everywhere (production keeps it until a v2-trained aligner ships); v2 must give
inference exactly the units the v2 manifest was built with. No model or checkpoint is loaded. The v2
pronunciation labels need phonemizer, espeak-ng and num2words (skipped where they are missing, e.g.
an analysis image built before the v2 rollout).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "fa_kara"
sys.path.insert(0, str(VENDOR))
import haruraw2norm as norm  # noqa: E402
import korean_pronunciation  # noqa: E402
import label_rules  # noqa: E402

V2_READY = all(importlib.util.find_spec(name) for name in ("phonemizer", "num2words"))
needs_v2 = pytest.mark.skipif(
    not V2_READY, reason="phonemizer/num2words (label rules v2) not installed"
)


def _lid_ready():
    try:
        label_rules.lid_model_path()
    except RuntimeError:
        return False
    return V2_READY and importlib.util.find_spec("fasttext") is not None


# Label rules v3 also need fastText and lid.176.ftz (/opt/fa_kara_models in the analysis image, or
# FA_KARA_LID_MODEL); skipped where they are missing, e.g. an image built before the v3 rollout.
needs_v3 = pytest.mark.skipif(
    not _lid_ready(), reason="fasttext / lid.176.ftz (label rules v3) not installed"
)


def prons(text, language, rules=None):
    items = (
        norm.process_haruhi_line(text, language, rules=rules)
        if rules
        else norm.process_haruhi_line(text, language)
    )
    return [i["pron"] for i in items if i.get("pron")]


def test_v1_is_the_default():
    for text, language in [
        ("I love you 1748", "auto"),
        ("待ってる", "ja"),
        ("मैं", "hi"),
        ("عن", "auto"),
        ("ñoño", "auto"),
        ("Çok güzel", "auto"),
        ("BFF", "en"),
    ]:
        assert norm.process_haruhi_line(text, language) == norm.process_haruhi_line(
            text, language, rules="v1"
        )
    assert label_rules.resolve(None) == label_rules.LabelRules("v1")
    with pytest.raises(ValueError):
        label_rules.resolve("v4")


def test_v2_script_fixes_without_g2p():
    assert prons("待ってる", "ja", "v2") == ["mat", "te", "ru"]
    assert prons("待ってる", "ja") == ["matsu", "te", "ru"]
    assert prons("मैं तुमसे", "hi", "v2") == ["main", "tumse"]
    assert prons("காதலின்", "auto", "v2") == ["kaadhalin"]
    assert prons("عن معك", "auto", "v2") == ["an", "mak"]
    assert prons("緣分", "yue", "v2") == ["yun", "fan"]
    assert prons("ñoño", "auto", "v2") == ["nyonyo"]
    assert korean_pronunciation.romanize_run("희망", True) == ["hi", "mang"]
    assert korean_pronunciation.romanize_run("희망") == ["hui", "mang"]


@needs_v2
def test_v2_pronunciation_labels():
    assert prons("knight through", "en", "v2") == ["nait", "thru"]
    assert prons("parlent beaucoup", "fr", "v2") == ["parl", "bo", "ku"]
    assert prons("Abend ich", "de", "v2") == ["abent", "ikh"]
    assert "".join(prons("1748", "en", "v2")) == "sevantinfortieit"
    assert prons("oh yeah", "en", "v2") == ["o", "yea"]
    for label in prons("Don't stop me now, 2 4 6 8 & 100%", "en", "v2"):
        assert label.isascii() and label.isalpha() and label.islower()


@needs_v2
def test_main_normalize_source_lines_uses_the_rules():
    import ast
    import unicodedata

    path = VENDOR / "main.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "normalize_source_lines"
    )
    scope = {"hn": norm, "unicodedata": unicodedata, "korean_pronunciation": korean_pronunciation}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    for language, line in [
        ("en", "In 1999 we were 21, ooh yeah"),
        ("fr", "J'avais ton âge"),
        ("de", "Ich hab immer gedacht"),
    ]:
        items, _ = scope["normalize_source_lines"]([line + "\n"], language, False, True, "v2")
        assert [i["pron"] for i in items if i.get("pron")] == prons(line, language, "v2")
        items, _ = scope["normalize_source_lines"]([line + "\n"], language, False, True)
        assert [i["pron"] for i in items if i.get("pron")] == prons(line, language)


def test_settings_and_language_forwarding(monkeypatch):
    from echora_analysis import karaoke_pipeline
    from echora_analysis.settings import Settings

    assert Settings().fa_kara_label_rules == "v3-sung"
    monkeypatch.setenv("FA_KARA_LABEL_RULES", "v4")
    with pytest.raises(ValueError):
        Settings()
    for rules in ("v3", "v3-orthographic"):
        monkeypatch.setenv("FA_KARA_LABEL_RULES", rules)
        assert Settings().fa_kara_label_rules == rules
    assert "en" not in karaoke_pipeline.fa_kara_languages("v1")
    assert {"en", "de", "fr", "tr", "yue", "zh"} <= karaoke_pipeline.fa_kara_languages("v2")
    # v3 also forwards the non-Latin-script languages it reads per line.
    assert karaoke_pipeline.fa_kara_languages("v2") < karaoke_pipeline.fa_kara_languages("v3")
    assert karaoke_pipeline.fa_kara_languages("v3-sung") == karaoke_pipeline.fa_kara_languages("v3")


@needs_v3
def test_v3_fixes():
    # before (v2) → after (v3): dropped g, lyric lexicon, letter spell-out, homographs.
    for text, before, after in [
        ("gettin", ["gen"], ["ge", "tin"]),
        ("tryna", ["tri", "na"], ["trai", "na"]),
        ("skrrt", ["eskeiarrarti"], ["skrt"]),
        ("live", ["laiv"], ["liv"]),
    ]:
        assert prons(text, "en", "v2") == before, text
        assert prons(text, "en", "v3") == after, text
    assert prons("close my eyes", "en", "v3")[0] == "kloz"
    assert prons("so close to you", "en", "v3")[1] == "klos"
    assert prons("il y a", "fr", "v3") == ["il", "i", "a"]
    # Per-line language: a Twi line of an English song is spelled, an English line of a German song is English.
    assert label_rules.line_language("Nso mayɛ dinn firi sɛ me nni bi mayɛ din", "en") == "auto"
    assert prons("Hey, I'm kinda glad you didn't pick up", "de", "v3")[:2] == ["hei", "aim"]
    assert prons("I live", "en", "v3-orthographic") == prons("I live", "en", "v2-orthographic")


@needs_v3
def test_v3_training_and_inference_units_identical():
    """main.py --label_rules v3 (FA_KARA_LABEL_RULES=v3 forwards en/de/fr) gives the units the v3
    manifest is built with: the manifest calls process_haruhi_line(line, hint, rules="v3") per line."""
    import ast
    import unicodedata

    path = VENDOR / "main.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "normalize_source_lines"
    )
    scope = {"hn": norm, "unicodedata": unicodedata, "korean_pronunciation": korean_pronunciation}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    lines = {
        "en": [
            "I'm tryna get it, gettin' closer, close my eyes",
            "Nso mayɛ dinn firi sɛ me nni bi mayɛ din",
            "Skrrt, 1999, R&B",
        ],
        "de": ["Hey, I'm kinda glad you didn't pick up", "Das ist 'n Ding"],
        "fr": ["Il y a des gens, j'y vais"],
    }
    for rules in ("v3", "v3-orthographic"):
        for language, texts in lines.items():
            items, _ = scope["normalize_source_lines"](
                [t + "\n" for t in texts], language, False, True, rules
            )
            training = [
                korean_pronunciation.apply_to_items(
                    norm.process_haruhi_line(t, language, rules=rules), v2=True
                )
                for t in texts
            ]
            assert [i["pron"] for i in items if i.get("pron")] == [
                i["pron"] for t in training for i in t if i.get("pron")
            ]
