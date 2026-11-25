"""Pronunciation-oriented labels for Devanagari, Gurmukhi and Tamil (label rules v2).

The v1 path hands these scripts to uroman, which transliterates the spelling: every anusvara is
``m`` (मैं → maim), the inherent schwa is never deleted (तुमसे → tumase), च is ``c``, ड़ is ``dd``, and
Tamil stops are never voiced (காதலின் → kaatalin). DATASET_AUDIT_2026-10-18.md §5 rated those as
high-severity for hi/pa (1.2 % of supervision) and medium for ta. v2 reads the script directly:

- Devanagari / Gurmukhi: table transliteration (long vowels doubled as before: aa, ii, uu; च → ch,
  छ → chh, ड़ → r, ढ़ → rh, nukta forms), anusvara / bindi / tippi as the homorganic nasal (ng before
  velars, m before labials, n otherwise and word-finally), candrabindu → n, visarga → h, addak doubles
  the next consonant, and schwa deletion: the inherent vowel is dropped word-finally and in the
  pattern VC_CV when the consonant is single (ज़िंदगी keeps its a after the nd cluster: zindagii; तुमसे
  → tumse, बचपन → bachpan, समझ → samajh, इश्क → ishk). In Gurmukhi a homorganic nasal does not block
  the deletion (ਲੰਘਦਾ → langhdaa).
- Tamil: table transliteration with the standard voicing of the stops க ச ட த ப ற: word-initial
  k/s/t/th/p/r, geminate kk/tch/tt/tth/pp/tr, after the homorganic nasal (ng)/j/d/dh/b/dr, between vowels
  g/s/d/dh/b/r (காதலின் → kaadhalin, சொல்லவா → sollavaa, தொட்டு → thottu); ழ and ள → l, ஞ → ny.

Every function returns a-z labels only; an unmapped character yields '' so the caller masks it.
"""

from __future__ import annotations

VERSION = "indic_v2"

_DEVANAGARI_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ऋ": "ri",
    "ॠ": "rii",
    "ऌ": "li",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "ऑ": "o",
    "ऍ": "e",
    "ऎ": "e",
    "ऒ": "o",
    "ॲ": "a",
}
_DEVANAGARI_SIGNS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ii",
    "ु": "u",
    "ू": "uu",
    "ृ": "ri",
    "ॄ": "rii",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ॉ": "o",
    "ॅ": "e",
    "ॆ": "e",
    "ॊ": "o",
    "ॢ": "li",
}
_DEVANAGARI_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ळ": "l",
    "क़": "k",
    "ख़": "kh",
    "ग़": "g",
    "ज़": "z",
    "ड़": "r",
    "ढ़": "rh",
    "फ़": "f",
    "य़": "y",
    "ऱ": "r",
    "ऩ": "n",
    "ॹ": "z",
    "ॸ": "d",
}
_DEVANAGARI_NUKTA = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "ज": "z",
    "ड": "r",
    "ढ": "rh",
    "फ": "f",
    "य": "y",
    "न": "n",
    "र": "r",
}

_GURMUKHI_VOWELS = {
    "ਅ": "a",
    "ਆ": "aa",
    "ਇ": "i",
    "ਈ": "ii",
    "ਉ": "u",
    "ਊ": "uu",
    "ਏ": "e",
    "ਐ": "ai",
    "ਓ": "o",
    "ਔ": "au",
}
_GURMUKHI_SIGNS = {
    "ਾ": "aa",
    "ਿ": "i",
    "ੀ": "ii",
    "ੁ": "u",
    "ੂ": "uu",
    "ੇ": "e",
    "ੈ": "ai",
    "ੋ": "o",
    "ੌ": "au",
}
_GURMUKHI_CONSONANTS = {
    "ਸ": "s",
    "ਹ": "h",
    "ਕ": "k",
    "ਖ": "kh",
    "ਗ": "g",
    "ਘ": "gh",
    "ਙ": "ng",
    "ਚ": "ch",
    "ਛ": "chh",
    "ਜ": "j",
    "ਝ": "jh",
    "ਞ": "ny",
    "ਟ": "t",
    "ਠ": "th",
    "ਡ": "d",
    "ਢ": "dh",
    "ਣ": "n",
    "ਤ": "t",
    "ਥ": "th",
    "ਦ": "d",
    "ਧ": "dh",
    "ਨ": "n",
    "ਪ": "p",
    "ਫ": "ph",
    "ਬ": "b",
    "ਭ": "bh",
    "ਮ": "m",
    "ਯ": "y",
    "ਰ": "r",
    "ਲ": "l",
    "ਵ": "v",
    "ੜ": "r",
    "ਸ਼": "sh",
    "ਖ਼": "kh",
    "ਗ਼": "g",
    "ਜ਼": "z",
    "ਫ਼": "f",
    "ਲ਼": "l",
}
_GURMUKHI_NUKTA = {"ਸ": "sh", "ਖ": "kh", "ਗ": "g", "ਜ": "z", "ਫ": "f", "ਲ": "l"}

# Place of articulation of a following consonant → nasal letter for anusvara / bindi / tippi.
_LABIAL = set("pbm")


def _nasal_for(next_label: str | None) -> str:
    """m before labials, n otherwise (n + k/g already spells the velar nasal: ਲੰਘਦਾ → langhdaa)."""
    if next_label and next_label[0] in _LABIAL:
        return "m"
    return "n"


def _brahmic_units(
    text: str,
    vowels: dict,
    signs: dict,
    consonants: dict,
    nukta_map: dict,
    virama: str,
    anusvara: set,
    candrabindu: set,
    visarga: str,
    nukta: str,
    addak: str,
) -> list[dict] | None:
    """Syllable units [{consonant, vowel, explicit, nasal, geminate}] in order; None if unmapped."""
    units: list[dict] = []
    geminate_next = False
    for index, char in enumerate(text):
        if char in consonants:
            label = consonants[char]
            if index + 1 < len(text) and text[index + 1] == nukta and char in nukta_map:
                label = nukta_map[char]
            units.append(
                {
                    "consonant": label,
                    "vowel": "a",
                    "explicit": False,
                    "nasal": "",
                    "geminate": geminate_next,
                }
            )
            geminate_next = False
        elif char == nukta:
            if not units:
                return None
        elif char in vowels:
            units.append(
                {
                    "consonant": "",
                    "vowel": vowels[char],
                    "explicit": True,
                    "nasal": "",
                    "geminate": False,
                }
            )
        elif char in signs:
            if not units or not units[-1]["consonant"]:
                return None
            units[-1]["vowel"], units[-1]["explicit"] = signs[char], True
        elif char == virama:
            if not units or not units[-1]["consonant"]:
                return None
            units[-1]["vowel"], units[-1]["explicit"] = "", True
        elif char in anusvara:
            if not units:
                return None
            units[-1]["nasal"] = "anusvara"
        elif char in candrabindu:
            if not units:
                return None
            units[-1]["nasal"] = "candrabindu"
        elif char == visarga:
            if not units:
                return None
            units[-1]["visarga"] = True
        elif char == addak:
            geminate_next = True
        elif char in "ऽ।॥":
            continue
        else:
            return None
    return units


def _schwa_deletion(units: list[dict], nasal_blocks: bool) -> None:
    """Drop inherent schwas word-finally and in VC_CV with a single consonant (right to left)."""
    if len(units) < 2:
        return
    last = units[-1]
    if last["consonant"] and not last["explicit"]:
        last["vowel"] = ""

    def has_vowel(unit):
        return bool(unit["vowel"])

    def single_consonant(index):
        previous = units[index - 1]
        if not has_vowel(previous):
            return False  # part of a conjunct (previous consonant carries a virama)
        if previous["nasal"] and nasal_blocks:
            return False  # anusvara before this consonant: a cluster (ज़िंदगी)
        return not units[index]["geminate"]

    for index in range(len(units) - 2, 0, -1):
        unit = units[index]
        if not unit["consonant"] or unit["explicit"] or not unit["vowel"]:
            continue
        if has_vowel(units[index - 1]) and has_vowel(units[index + 1]) and single_consonant(index):
            unit["vowel"] = ""


def _render(units: list[dict]) -> str:
    out = []
    for index, unit in enumerate(units):
        consonant = unit["consonant"]
        if unit["geminate"] and consonant:
            out.append(consonant[0])
        out.append(consonant)
        out.append(unit["vowel"])
        if unit["nasal"]:
            following = (
                next((u["consonant"] for u in units[index + 1 :] if u["consonant"]), None)
                if index + 1 < len(units)
                else None
            )
            out.append(_nasal_for(following))
        if unit.get("visarga"):
            out.append("h")
    return "".join(out)


def devanagari_label(text: str) -> str:
    units = _brahmic_units(
        text,
        _DEVANAGARI_VOWELS,
        _DEVANAGARI_SIGNS,
        _DEVANAGARI_CONSONANTS,
        _DEVANAGARI_NUKTA,
        virama="्",
        anusvara={"ं"},
        candrabindu={"ँ"},
        visarga="ः",
        nukta="़",
        addak="",
    )
    if units is None:
        return ""
    _schwa_deletion(units, nasal_blocks=True)
    return _render(units)


def gurmukhi_label(text: str) -> str:
    units = _brahmic_units(
        text,
        _GURMUKHI_VOWELS,
        _GURMUKHI_SIGNS,
        _GURMUKHI_CONSONANTS,
        _GURMUKHI_NUKTA,
        virama="੍",
        anusvara={"ਂ", "ੰ"},
        candrabindu=set(),
        visarga="ਃ",
        nukta="਼",
        addak="ੱ",
    )
    if units is None:
        return ""
    _schwa_deletion(units, nasal_blocks=False)
    return _render(units)


# ---------------------------------------------------------------------------------------------
# Tamil

_TAMIL_VOWELS = {
    "அ": "a",
    "ஆ": "aa",
    "இ": "i",
    "ஈ": "ii",
    "உ": "u",
    "ஊ": "uu",
    "எ": "e",
    "ஏ": "ee",
    "ஐ": "ai",
    "ஒ": "o",
    "ஓ": "oo",
    "ஔ": "au",
}
_TAMIL_SIGNS = {
    "ா": "aa",
    "ி": "i",
    "ீ": "ii",
    "ு": "u",
    "ூ": "uu",
    "ெ": "e",
    "ே": "ee",
    "ை": "ai",
    "ொ": "o",
    "ோ": "oo",
    "ௌ": "au",
}
_TAMIL_CONSONANTS = {
    "க": "k",
    "ங": "ng",
    "ச": "s",
    "ஞ": "ny",
    "ட": "t",
    "ண": "n",
    "த": "th",
    "ந": "n",
    "ப": "p",
    "ம": "m",
    "ய": "y",
    "ர": "r",
    "ல": "l",
    "வ": "v",
    "ழ": "l",
    "ள": "l",
    "ற": "r",
    "ன": "n",
    "ஜ": "j",
    "ஷ": "sh",
    "ஸ": "s",
    "ஹ": "h",
}
# stop: (initial, geminate, after homorganic nasal, between vowels / after other consonants)
_TAMIL_STOPS = {
    "க": ("k", "kk", "", "g", "k"),
    "ச": ("s", "tch", "j", "s", "ch"),
    "ட": ("d", "tt", "d", "d", "t"),
    "த": ("th", "tth", "dh", "dh", "th"),
    "ப": ("p", "pp", "b", "b", "p"),
    "ற": ("r", "tr", "dr", "r", "r"),
}
_TAMIL_NASAL_OF = {"க": "ங", "ச": "ஞ", "ட": "ண", "த": "ந", "ப": "ம", "ற": "ன"}
_PULLI = "்"


def tamil_label(text: str) -> str:
    units: list[dict] = []  # {char, consonant, vowel}
    for char in text:
        if char in _TAMIL_CONSONANTS:
            units.append({"char": char, "vowel": "a"})
        elif char in _TAMIL_VOWELS:
            units.append({"char": "", "vowel": _TAMIL_VOWELS[char]})
        elif char in _TAMIL_SIGNS:
            if not units or not units[-1]["char"]:
                return ""
            units[-1]["vowel"] = _TAMIL_SIGNS[char]
        elif char == _PULLI:
            if not units or not units[-1]["char"]:
                return ""
            units[-1]["vowel"] = ""
        elif char == "ஃ":
            units.append({"char": "", "vowel": "", "extra": "h"})
        else:
            return ""
    out = []
    for index, unit in enumerate(units):
        char = unit["char"]
        if char in _TAMIL_STOPS:
            initial, geminate, nasal, voiced, voiceless = _TAMIL_STOPS[char]
            previous = units[index - 1] if index else None
            if previous is None:
                label = initial
            elif previous["char"] == char and not previous["vowel"]:
                label = ""  # the second half of a geminate: emitted with the first
            elif previous["char"] == _TAMIL_NASAL_OF[char] and not previous["vowel"]:
                label = nasal
            elif previous["vowel"] or not previous["char"]:
                label = voiced
            else:
                label = voiceless
            following = units[index + 1] if index + 1 < len(units) else None
            if not unit["vowel"] and following is not None and following["char"] == char:
                label = geminate
            out.append(label)
        elif char:
            out.append(_TAMIL_CONSONANTS[char])
        out.append(unit["vowel"])
        out.append(unit.get("extra", ""))
    return "".join(out)
