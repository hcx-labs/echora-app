"""Pronunciation-oriented romanization for scripts the upstream FA-Kara normalizer did not cover.

Every helper returns normalizer items ``{'orig', 'type', 'pron'}`` whose ``orig`` values
concatenate exactly to the input run, with ``pron`` in lowercase ``a-z`` (or empty when the
unit is silent or unknown; the caller masks those). Labels are acoustic spellings in FA-Kara's
30-letter alphabet, not a standard transliteration: the same letters have to mean roughly the
same sounds across languages because one CTC head serves all of them.

Methods and licences (all deterministic, no network):

- Mandarin: ``pypinyin`` (MIT) phrase-level readings after OpenCC (Apache-2.0) ``t2s``, so
  traditional lyrics get phrase disambiguation (銀行 → yin hang, not yin xing). Tones dropped,
  pinyin respelled by sound (``mandarin_v1``: q→ch, x→sh, zh→j, c→ts, ü→yu, apical zhi/chi/shi/ri
  → jr/chr/shr/r, iu→iou, ui→uei, un→uen, ian→ien).
- Cantonese: ``ToJyutping`` (BSD-2-Clause) character readings; tones dropped, j→y, c→ts.
- Thai: ``pythainlp`` (Apache-2.0) dictionary syllable segmentation (Apple times Thai per
  syllable) and Royal Thai General System romanization per syllable, with a small exception
  table where the rule engine leaves Thai marks behind and ``uroman`` as the last resort.
- Arabic script: ``uroman`` (MIT) consonantal transliteration, then ي/و between consonants are
  read as the long vowels i/u and the ʻayn/hamza apostrophe is dropped. Short vowels are not
  written in lyrics, so labels stay vowel-poor; see MULTILINGUAL_NORMALIZATION.md.
- Every other letter script (Cyrillic, Greek, Hebrew, Tamil, Telugu, Bengali, ...): ``uroman``
  with the script's language code. Devanagari and Gurmukhi keep the legacy path unchanged.
- Latin: Turkish letters in a line (ı İ ş ğ) switch ``uroman`` to its Turkish rules (ç → ch).
"""

from __future__ import annotations

import re
import unicodedata

VERSION = (
    "scripts_v1"  # v2 behaviour is selected per call (label_rules), the v1 output is unchanged
)

_LETTERS = re.compile(r"[^a-z]")

# ---------------------------------------------------------------------------------------------
# Mandarin

_MANDARIN_INITIALS = {
    "b": "b",
    "p": "p",
    "m": "m",
    "f": "f",
    "d": "d",
    "t": "t",
    "n": "n",
    "l": "l",
    "g": "g",
    "k": "k",
    "h": "h",
    "j": "j",
    "q": "ch",
    "x": "sh",
    "zh": "j",
    "ch": "ch",
    "sh": "sh",
    "r": "r",
    "z": "z",
    "c": "ts",
    "s": "s",
}
# Finals after an initial. ``v`` is pypinyin's ü.
_MANDARIN_FINALS = {
    "a": "a",
    "o": "o",
    "e": "e",
    "i": "i",
    "u": "u",
    "v": "yu",
    "ê": "e",
    "ai": "ai",
    "ei": "ei",
    "ao": "ao",
    "ou": "ou",
    "an": "an",
    "en": "en",
    "ang": "ang",
    "eng": "eng",
    "ong": "ong",
    "er": "er",
    "ia": "ia",
    "ie": "ie",
    "iao": "iao",
    "iu": "iou",
    "ian": "ien",
    "in": "in",
    "iang": "iang",
    "ing": "ing",
    "iong": "iong",
    "ua": "ua",
    "uo": "uo",
    "uai": "uai",
    "ui": "uei",
    "uan": "uan",
    "un": "uen",
    "uang": "uang",
    "ueng": "ueng",
    "ve": "yue",
    "van": "yuen",
    "vn": "yun",
}
# Whole syllables without an initial (pinyin already writes the glide) and the apical vowels.
_MANDARIN_SYLLABLES = {
    "yi": "i",
    "ya": "ya",
    "yan": "yen",
    "yang": "yang",
    "yao": "yao",
    "ye": "ye",
    "yin": "in",
    "ying": "ing",
    "yo": "yo",
    "yong": "yong",
    "you": "you",
    "yu": "yu",
    "yuan": "yuen",
    "yue": "yue",
    "yun": "yun",
    "wu": "u",
    "wa": "wa",
    "wai": "wai",
    "wan": "wan",
    "wang": "wang",
    "wei": "wei",
    "wen": "wen",
    "weng": "weng",
    "wo": "wo",
    "zhi": "jr",
    "chi": "chr",
    "shi": "shr",
    "ri": "r",
    "zi": "zi",
    "ci": "tsi",
    "si": "si",
    "m": "m",
    "n": "n",
    "ng": "ng",
    "hm": "hm",
    "hng": "hng",
}
_JQX_FINALS = {"u": "yu", "ue": "yue", "uan": "yuen", "un": "yun"}


def pinyin_to_label(pinyin: str) -> str:
    """Tone-less acoustic spelling of one pypinyin ``TONE3`` syllable; '' when unknown."""
    syllable = re.sub(r"[0-5]", "", pinyin.lower()).replace("ü", "v")
    if not syllable:
        return ""
    if syllable in _MANDARIN_SYLLABLES:
        return _MANDARIN_SYLLABLES[syllable]
    initial = next(
        (i for i in ("zh", "ch", "sh", *"bpmfdtnlgkhjqxrzcs") if syllable.startswith(i)), ""
    )
    final = syllable[len(initial) :]
    if not initial:
        return _MANDARIN_FINALS.get(final, "") if final and final[0] in "aoeê" else ""
    if initial in ("j", "q", "x") and final in _JQX_FINALS:
        return _MANDARIN_INITIALS[initial] + _JQX_FINALS[final]
    label = _MANDARIN_FINALS.get(final)
    return _MANDARIN_INITIALS[initial] + label if label else ""


def _simplified(text: str) -> str:
    try:
        import opencc
    except ImportError as error:  # pragma: no cover - dependency documented in pyproject
        raise RuntimeError("Mandarin normalization needs the OpenCC package (t2s)") from error
    converter = _simplified.__dict__.setdefault("converter", opencc.OpenCC("t2s"))
    converted = converter.convert(text)
    if len(converted) != len(text):
        converted = "".join(
            converter.convert(char) if len(converter.convert(char)) == 1 else char for char in text
        )
    return converted if len(converted) == len(text) else text


def mandarin_readings(text: str) -> list[str]:
    """One label per character of a Han run (phrase-level heteronym choice), '' when unknown."""
    from pypinyin import Style, pinyin

    readings = pinyin(_simplified(text), style=Style.TONE3, heteronym=False)
    if len(readings) != len(text):
        return [""] * len(text)
    return [pinyin_to_label(item[0]) if item else "" for item in readings]


# ---------------------------------------------------------------------------------------------
# Cantonese

# Characters that are (almost) only written in Cantonese: a line containing one is read in
# Jyutping under ``auto``. Standard-written-Chinese Cantopop needs the explicit ``yue`` hint.
CANTONESE_MARKERS = frozenset("嘅唔佢咗哋嘢冇噉啲乜嗰喺畀攞咁嚟啱嘥瞓諗咩搵")


def jyutping_to_label(jyutping: str, v2: bool = False) -> str:
    syllable = re.sub(r"[0-9]", "", jyutping.lower())
    if v2:
        # Jyutping writes /jy/ as "jyu" (雨 jyu5, 月 jyut6): one glide, not "yyu" (audit §5 #8).
        syllable = re.sub(r"^jyu", "yu", syllable)
    syllable = re.sub(r"^j", "y", syllable)
    syllable = re.sub(r"^c", "ts", syllable)
    return syllable if syllable and not _LETTERS.search(syllable) else ""


def cantonese_readings(text: str, v2: bool = False) -> list[str]:
    try:
        import ToJyutping
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Cantonese normalization needs the ToJyutping package") from error
    pairs = ToJyutping.get_jyutping_list(text)
    if len(pairs) != len(text):
        return [""] * len(text)
    return [jyutping_to_label(reading, v2) if reading else "" for _, reading in pairs]


# ---------------------------------------------------------------------------------------------
# Thai

# Syllables where pythainlp's rule engine leaves a Thai mark behind, or reads a frequent
# function word wrongly.
_THAI_EXCEPTIONS = {
    "ก็": "ko",
    "เพราะ": "phro",
    "เอ๊ะ": "e",
    "อ": "o",
    "เหน็บ": "nep",
    "เถอะ": "tho",
    "นะ": "na",
    "ล่ะ": "la",
    "เนอะ": "noe",
    "เอะ": "e",
    "ค่ะ": "kha",
    "คะ": "kha",
}
_THAI_LETTER = re.compile(r"[฀-๿]")


def _thai_syllable_label(syllable: str, uroman) -> str:
    if syllable in _THAI_EXCEPTIONS:
        return _THAI_EXCEPTIONS[syllable]
    from pythainlp.transliterate import romanize

    label = romanize(syllable, engine="royin").lower()
    if label and not _LETTERS.search(label):
        return label
    label = uroman.romanize_string(syllable).lower().replace("-", "").replace(" ", "")
    return label if label and not _LETTERS.search(label) else ""


def thai_items(text: str, uroman) -> list[dict]:
    """Per-syllable items; ๆ repeats the previous syllable; unknown syllables stay unlabelled."""
    try:
        from pythainlp.tokenize import subword_tokenize
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Thai normalization needs the pythainlp package") from error
    items: list[dict] = []
    previous = ""
    for syllable in subword_tokenize(text, engine="dict", keep_whitespace=True):
        if not syllable:
            continue
        if syllable == "ๆ":
            items.append({"orig": syllable, "type": 7, "pron": previous})
            continue
        if not _THAI_LETTER.search(syllable) or not any(
            unicodedata.category(c)[0] in "LM" for c in syllable
        ):
            items.append({"orig": syllable, "type": 0})
            continue
        label = _thai_syllable_label(syllable, uroman)
        items.append(
            {"orig": syllable, "type": 7, "pron": label} if label else {"orig": syllable, "type": 0}
        )
        previous = label or previous
    if "".join(item["orig"] for item in items) != text:
        return [{"orig": text, "type": 0}]
    return items


# ---------------------------------------------------------------------------------------------
# Arabic script and the generic uroman path

_CONSONANTS = set("bcdfghjklmnpqrstvxz")


_ARABIC_DIGRAPHS = ("sh", "th", "kh", "gh", "dh")


def arabic_label(text: str, uroman, v2: bool = False) -> str:
    """uroman, then ي/و between consonants as the long vowels i/u; ʻayn/hamza marks dropped.

    v2 (audit §5 #5): ʿayn/hamza read as the vowel they carry (عن → an, معك → mak, عندي → andi) and the
    unwritten short vowel is restored as ``a`` inside word-initial and three-consonant clusters and
    before a final ه (حبيبي → habibi, الله → allah, قلب → qalb); doubled consonants (shadda) stay.
    """
    raw = uroman.romanize_string(text).lower().replace("-", "").replace(" ", "")
    raw = raw.replace("'", "a" if v2 else "")
    if v2:
        raw = re.sub(r"aa+", "a", raw)
    letters = list(raw)
    for index, char in enumerate(raw):
        if char not in "yw":
            continue
        before = raw[index - 1] if index else ""
        after = raw[index + 1] if index + 1 < len(raw) else ""
        if before in _CONSONANTS and (after in _CONSONANTS or not after):
            letters[index] = "i" if char == "y" else "u"
    label = "".join(letters)
    return _arabic_skeleton_vowels(label) if v2 else label


def _arabic_skeleton_vowels(label: str) -> str:
    """Insert ``a`` into consonant clusters a sung Arabic word cannot have (heuristic, see above)."""
    tokens = []
    position = 0
    while position < len(label):
        if label.startswith(_ARABIC_DIGRAPHS, position):
            tokens.append(label[position : position + 2])
            position += 2
        else:
            tokens.append(label[position])
            position += 1
    consonant = [t[0] in _CONSONANTS for t in tokens]
    out = []
    for index, token in enumerate(tokens):
        out.append(token)
        if index + 1 >= len(tokens) or not consonant[index] or not consonant[index + 1]:
            continue
        nxt = tokens[index + 1]
        if nxt == token:
            continue  # shadda
        word_initial = index == 0
        third = index + 2 < len(tokens) and consonant[index + 2]
        final_h = nxt == "h" and index + 2 == len(tokens)
        if word_initial or third or final_h:
            out.append("a")
    return "".join(out)


# uroman language codes per Unicode script (first word of the character name).
UROMAN_SCRIPTS = {
    "CYRILLIC": "rus",
    "GREEK": "ell",
    "HEBREW": "heb",
    "ARMENIAN": "hye",
    "GEORGIAN": "kat",
    "BENGALI": "ben",
    "GUJARATI": "guj",
    "KANNADA": "kan",
    "MALAYALAM": "mal",
    "TAMIL": "tam",
    "TELUGU": "tel",
    "SINHALA": "sin",
    "KHMER": "khm",
    "LAO": "lao",
    "MYANMAR": "mya",
    "ETHIOPIC": "amh",
    "TIBETAN": "bod",
    "ORIYA": "ori",
    "MONGOLIAN": "mon",
    "THAANA": "div",
    "SYRIAC": "syr",
    "TIFINAGH": "ber",
    "CHEROKEE": "chr",
    "CANADIAN": "cr",
}


def script_name(char: str) -> str:
    return unicodedata.name(char, "").split(" ", 1)[0]


def uroman_label(text: str, uroman, lcode: str | None = None) -> str:
    label = uroman.romanize_string(text, lcode=lcode) if lcode else uroman.romanize_string(text)
    return label.lower().replace("-", "").replace(" ", "").replace("'", "")


# ---------------------------------------------------------------------------------------------
# Latin

_TURKISH_LETTERS = set("ıİşŞğĞ")


def latin_lcode(line: str, lang: str) -> str | None:
    """uroman language rules for Latin runs: Turkish when the hint or the letters say so."""
    if lang == "tr" or any(char in _TURKISH_LETTERS for char in line):
        return "tur"
    return None


_HORN = "\u031b"


def latin_v2_source(text: str, lcode: str | None) -> str:
    """Label rules v2 respellings applied before uroman (DATASET_AUDIT_2026-10-18.md §5 #9, #11).

    - ñ → ny (Spanish/Portuguese; uroman folded it to n: ñoño → nono).
    - Turkish (``lcode == "tur"``): ğ is silent and lengthens the vowel before it (değil → deil,
      dağ → da), so it is dropped instead of read g.
    - Vietnamese ư / ơ (with any tone mark) keep a distinct label: ư → uw, ơ → ow (Telex spelling;
      uroman folded them into u / o, merging người with nguoi). â/ă/ê/ô still fold (their quality is
      close to a/a/e/o) and đ/d both stay d (the d reading differs by dialect: z north, y south).
    """
    source = text.replace("ñ", "ny").replace("Ñ", "Ny")
    if lcode == "tur":
        source = source.replace("ğ", "").replace("Ğ", "")
    if _HORN in unicodedata.normalize("NFD", source):
        decomposed = unicodedata.normalize("NFD", source)
        decomposed = (
            decomposed.replace("u" + _HORN, "uw")
            .replace("U" + _HORN, "Uw")
            .replace("o" + _HORN, "ow")
            .replace("O" + _HORN, "Ow")
        )
        source = unicodedata.normalize("NFC", decomposed)
    return source
