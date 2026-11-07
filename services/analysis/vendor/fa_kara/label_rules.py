"""Versioned label rules for the shared FA-Kara normalizer (``process_haruhi_line(..., rules=)``).

``v1`` (default everywhere) is today's behaviour byte for byte: Latin runs are spelled (uroman),
digits go through the CMU-romaji number path, acronyms are spelled, vocables keep their silent ``h``.
Production and every manifest built before 2026-10-18 use it.

``v2`` (DATASET_AUDIT_2026-10-18.md §3, §5; DATASET_FIXES_2026-10-18.md) changes the labels of
Latin-script lines and is selected explicitly (training: ``build_karaoke_manifest.py --label-rules
v2``; inference: ``main.py --label_rules v2`` / ``FA_KARA_LABEL_RULES=v2``), so production keeps v1
until a model trained on v2 labels ships, as the Korean rollout did:

- **Pronunciation labels for English, German and French** (user decision 2026-10-18) when the
  language hint is ``en``/``de``/``fr``: dictionary-driven G2P plus rules for out-of-vocabulary words,
  respelled into the a-z alphabet by sound (``knight`` → nait, ``parlent`` → parl, ``beaucoup`` →
  boku, ``Abend`` → abent, ``ich`` → ikh). English in-vocabulary words come from CMUdict (nltk,
  BSD-2-Clause); everything else (English OOV, all German and French) from espeak-ng through
  phonemizer (both GPL-3.0; combined with Echora's AGPL-3.0 under GPLv3 §13). One shared IPA → letter
  table keeps the same sounds on the same letters across the three languages (th/dh/sh/zh/ch/j/ng/kh,
  ai/au/ei/oi, French nasals an/on/en/un, ü → yu, ö/eu → oe). A word splits into syllables only when
  pyphen's surface syllables and the phone syllables agree in number (the same "only where the
  mapping is unambiguous" rule the Japanese path uses); otherwise the word stays one item and an Apple
  fragment boundary inside it is masked by the timed adapter. ``v2-orthographic`` keeps v2's other
  fixes with spelled Latin labels (the one-variable control).
- **Numbers** are read as words in the hint's language (num2words, LGPL-2.1; years, ordinals,
  leading zeros, thousands separators) and then labelled like words — never through the retired
  CMU-romaji map (``1748`` was ``wansauzandsevanhandradandfortieit``).
- **Acronyms** (2-4 capitals without a vowel, or a short list of common ones) are letter names.
- **Vocables / ad-libs** use a small lexicon of sung forms (oh → o, ooh → u, yeah → yea, whoa → wo,
  mm → m, hey → hei) instead of their spelling with a silent ``h``.
- **Symbols** ``& % $ + @ € £`` are read (``R&B`` → ar and bi; ``$5`` → five dollars).
- Spanish/Portuguese ``ñ`` → ny; Turkish lines use uroman's Turkish rules through the ``tr`` hint.
- The curly apostrophe ’ is part of a word like ' (``don’t`` is one G2P lookup).

``v3`` (data/finetune-curation/LABEL_RULES_V3.md) is v2 plus fixes measured on the v2 training set;
v1 and v2 output is unchanged. ``v3-orthographic`` is v3 with spelled en/de/fr words (the A/B control,
like ``v2-orthographic``):

- **Dropped-g elisions** (``gettin``, ``nothin'``) read the CMUdict ``-ing`` entry with the final NG
  sung as n (``gettin`` was espeak's ``gen``; ``nothin'`` was ``na thing``).
- **Lyric lexicon** (``SLANG_EN``): sung forms of casual spellings (tryna → traina, kinda → kainda,
  outta → auta, imma → aima, gyal → gyal, …).
- **Letter spell-outs**: when espeak (or a CMUdict acronym entry) reads a lower-case word letter by
  letter (``skrrt`` → eskeiarrarti, ``gbe`` → jibii), the word's own letters are used, repeats collapsed.
- **Homographs** (``HOMOGRAPHS_EN``): the reading most common in lyrics instead of CMUdict's first
  (live → liv, wind → wind, lead → lid, tears → tirz), with a neighbouring-word cue where one decides
  it (``close my eyes`` → kloz, ``wind up`` → waind, ``no use`` → yus, ``used to`` → yust).
- **Per-line language** (``line_language``): fastText lid.176 on each Latin line of an en/de/fr song;
  a line confidently in another language goes to that language's normaliser (de/fr pronunciation) or
  the generic Latin path, never English G2P; an English line in a de/fr song goes to English G2P.
- German/French: English contractions and words espeak itself switches to English use the English
  path; French ``y`` reads i (was the letter name igrek), German ``'n`` reads n; the v2 crash on a
  vocable after a French clitic (``j'm``) is fixed.

``v3-sung`` (opt-in, 2026-10-29) is v3 with sung unstressed vowels for English: a reduced vowel
(CMUdict AH0, espeak ə / ᵻ) takes the vowel letter it is spelled with, found by aligning the phones
to the word's letters (``feminine`` F EH1 M AH0 N AH0 N → feminin, not femanan; ``spirit`` →
spirit), and among CMUdict's readings the one that aligns best with the spelling is used instead
of always the first. A reduced vowel without exactly one spelled vowel letter keeps v3's label. A
word whose pyphen syllables disagree in number with its phone syllables is cut through the same
alignment (intuition → in|tu|i|tion) instead of staying one item.

Everything here is deterministic and offline; the espeak-ng version is recorded by the callers.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import unicodedata

VERSION = "label_rules_v2"
VERSION_V3 = "label_rules_v3_r2"  # forwarding, apostrophe eligibility, contiguous foreign runs
VERSION_V3_SUNG = "label_rules_v3_r2+sung_vowels_r1"
RULES = ("v1", "v2", "v2-orthographic", "v3", "v3-orthographic", "v3-sung")
PRONUNCIATION_LANGUAGES = frozenset({"en", "de", "fr"})
# Languages whose hint v2 understands beyond v1's (ja, jaen, zhen, ko, ur, hi, pa, indic, yue, zh).
V2_HINTS = frozenset({"en", "de", "fr", "tr"})

logger = logging.getLogger(__name__)


class LabelRules:
    """Immutable description of one rule set."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        if name not in RULES:
            raise ValueError(f"label rules must be one of {RULES}, not {name!r}")
        self.name = name

    @property
    def v2(self) -> bool:
        return self.name != "v1"

    @property
    def v3(self) -> bool:
        return self.name.startswith("v3")

    @property
    def pronunciation(self) -> frozenset:
        return PRONUNCIATION_LANGUAGES if self.name in ("v2", "v3", "v3-sung") else frozenset()

    @property
    def sung_vowels(self) -> bool:
        return self.name == "v3-sung"

    @property
    def version(self) -> str:
        """The label-rules part of the normalizer version (v2 keeps its historical string)."""
        return VERSION_V3_SUNG if self.sung_vowels else VERSION_V3 if self.v3 else VERSION

    def __repr__(self) -> str:
        return f"LabelRules({self.name!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, LabelRules) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


def resolve(rules) -> LabelRules:
    if rules is None:
        return LabelRules("v1")
    if isinstance(rules, LabelRules):
        return rules
    return LabelRules(str(rules))


# ---------------------------------------------------------------------------------------------
# Shared sound alphabet

# IPA (espeak-ng output) → label letters. Longest match first; vowels carry kind "V".
IPA_MAP = {
    # diphthongs and nasal vowels first (longest match)
    "aɪə": "aia",
    "aʊə": "aua",
    "aɪ": "ai",
    "aʊ": "au",
    "eɪ": "ei",
    "oʊ": "o",
    "əʊ": "o",
    "ɔɪ": "oi",
    "ɔʏ": "oi",
    "ɔø": "oi",
    "ɛə": "e",
    "ɪə": "ia",
    "ʊə": "ua",
    "eə": "e",
    "ɑ̃": "an",
    "ɔ̃": "on",
    "ɛ̃": "en",
    "œ̃": "un",
    "ɑ̃ː": "an",
    # vowels
    "a": "a",
    "ɑ": "a",
    "æ": "a",
    "ʌ": "a",
    "ɐ": "a",
    "ə": "a",
    "ɚ": "er",
    "ɝ": "er",
    "ɜ": "er",
    "e": "e",
    "ɛ": "e",
    "ɪ": "i",
    "i": "i",
    "ɨ": "i",
    "ɔ": "o",
    "o": "o",
    "ɒ": "o",
    "ʊ": "u",
    "u": "u",
    "ʉ": "u",
    "y": "yu",
    "ʏ": "yu",
    "ø": "oe",
    "œ": "oe",
    # consonants
    "tʃ": "ch",
    "dʒ": "j",
    "ts": "ts",
    "dz": "dz",
    "pf": "pf",
    "b": "b",
    "d": "d",
    "f": "f",
    "g": "g",
    "ɡ": "g",
    "h": "h",
    "k": "k",
    "l": "l",
    "ɫ": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "s": "s",
    "t": "t",
    "v": "v",
    "w": "w",
    "z": "z",
    "ɹ": "r",
    "r": "r",
    "ʁ": "r",
    "ɾ": "r",
    "ɻ": "r",
    "ʀ": "r",
    "θ": "th",
    "ð": "dh",
    "ʃ": "sh",
    "ʒ": "zh",
    "ŋ": "ng",
    "j": "y",
    "ɲ": "ny",
    "ç": "kh",
    "x": "kh",
    "χ": "kh",
    "ʔ": "",
    "ʎ": "ly",
    "ɥ": "w",
    "ʍ": "w",
    "ɬ": "l",
    "ʋ": "v",
    "β": "v",
    "ɣ": "g",
    "ɵ": "oe",
    "ɯ": "u",
    "ɘ": "e",
}
IPA_VOWELS = frozenset(k for k, v in IPA_MAP.items() if k and k[0] in "aɑæʌɐəɚɝɜeɛɪiɨɔoɒʊuʉyʏøœɵɯɘ")
# Per-language overrides (the same sound sits on the same letter where the languages share it;
# these are the symbols espeak uses differently per voice).
IPA_OVERRIDES = {
    "en": {"ɾ": "t", "ə": "a"},  # flapped t stays t (as CMUdict's T); schwa as in CMU AH0
    "de": {"ə": "e", "ɜ": "a", "ɐ": "a"},  # vocalic -er ending reads a (Mutter → muta)
    "fr": {"ə": "e"},
}
_IPA_IGNORED = set("ːˈˌ͡‿.̩̯͡ʰ̃ -_")
_IPA_KEYS = sorted({*IPA_MAP, "ᵻ"}, key=len, reverse=True)  # ᵻ is read only as a v3-sung slot

# CMUdict ARPAbet → the same letters (v2; the v1 map with L→r and TH→s is in haruraw2norm).
ARPABET_MAP = {
    "AA": "a",
    "AE": "a",
    "AH": "a",
    "AO": "o",
    "AW": "au",
    "AY": "ai",
    "B": "b",
    "CH": "ch",
    "D": "d",
    "DH": "dh",
    "EH": "e",
    "ER": "er",
    "EY": "ei",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "i",
    "IY": "i",
    "JH": "j",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "OW": "o",
    "OY": "oi",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "sh",
    "T": "t",
    "TH": "th",
    "UH": "u",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "zh",
}
ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)

ESPEAK_VOICES = {"en": "en-us", "de": "de", "fr": "fr-fr"}
PYPHEN_LANGUAGES = {"en": "en_US", "de": "de_DE", "fr": "fr_FR"}


def ipa_tokens(ipa: str, language: str, reduced: bool = False) -> list[tuple[str, bool]]:
    """[(label letters, is_vowel)] from an espeak IPA string; unknown symbols are dropped.

    ``reduced`` (v3-sung, English): ə and ᵻ become the slots ``@a`` / ``@i`` for ``sung_vowels``."""
    overrides = {**IPA_OVERRIDES.get(language, {}), **({"ə": "@a", "ᵻ": "@i"} if reduced else {})}
    text = (
        ipa  # espeak writes nasal vowels with the combining tilde; ç and the like stay precomposed
    )
    tokens: list[tuple[str, bool]] = []
    position = 0
    while position < len(text):
        char = text[position]
        if char in _IPA_IGNORED:
            position += 1
            continue
        for key in _IPA_KEYS:
            if text.startswith(key, position):
                label = overrides.get(key, IPA_MAP.get(key, ""))
                if (
                    language == "fr"
                    and key == "y"
                    and position + 1 < len(text)
                    and text[position + 1] in "aeiouɛɔəɑ"
                ):
                    label = (
                        "w"  # espeak-fr writes the glide of lui / nuit / huit as y before a vowel
                    )
                    tokens.append((label, False))
                elif label:
                    tokens.append((label, key in IPA_VOWELS or label.startswith("@")))
                position += len(key)
                break
        else:
            if char.isascii() and char.isalpha():
                tokens.append((char.lower(), char.lower() in "aeiou"))
            position += 1
    return tokens


def arpabet_tokens(phones: list[str]) -> list[tuple[str, bool]]:
    tokens = []
    for phone in phones:
        base = phone.rstrip("012")
        label = ARPABET_MAP.get(base, "")
        if label:
            tokens.append((label, base in ARPABET_VOWELS))
    return tokens


def syllable_cuts(tokens: list[tuple[str, bool]]) -> list[int]:
    """Token indices where phone syllables after the first start (see ``syllabify``)."""
    nuclei = [index for index, (_, vowel) in enumerate(tokens) if vowel]
    return [
        right - 1 if right - left - 1 >= 1 else right for left, right in zip(nuclei, nuclei[1:])
    ]


def syllabify(tokens: list[tuple[str, bool]]) -> list[str]:
    """Phone syllables: one per vowel nucleus; a consonant cluster between two nuclei gives its
    last consonant to the next syllable (maximal-onset simplification)."""
    if sum(1 for _, vowel in tokens if vowel) <= 1:
        return ["".join(label for label, _ in tokens)] if tokens else []
    cuts = syllable_cuts(tokens)
    pieces, start = [], 0
    for cut in cuts + [len(tokens)]:
        pieces.append("".join(label for label, _ in tokens[start:cut]))
        start = cut
    return pieces


# ---------------------------------------------------------------------------------------------
# Lexicons

# Sung forms of vocables and ad-libs (DATASET_AUDIT §3: the spelling put a silent h on the vowel's
# decay). Matched on the lower-cased word with repeated letters collapsed to two.
VOCABLES: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern), label)
    for pattern, label in [
        (r"^oo+h*$", "u"),
        (r"^o+h+$", "o"),
        (r"^a+h+$", "a"),
        (r"^aa+$", "a"),
        (r"^u+h+$", "a"),
        (r"^e+h+$", "e"),
        (r"^(y+e+a+h*|y+e+h+|yea+)$", "yea"),
        (r"^wh?o+a+h*$", "wo"),
        (r"^who+$", "wu"),
        (r"^woo+h*$", "wu"),
        (r"^hoo+$", "hu"),
        (r"^m+h*m*$", "m"),
        (r"^h+m+$", "hm"),
        (r"^mhm+$", "mhm"),
        (r"^he+y+$", "hei"),
        (r"^ay+e*$", "ai"),
        (r"^s+h+$", "sh"),
        (r"^doo+$", "du"),
        (r"^ooo+$", "u"),
        (r"^oh+o+$", "oo"),
        (r"^uh+-?huh+$", "aha"),
        (r"^ha+h$", "ha"),
        (r"^ya+h$", "ya"),
        (r"^yo+h*$", "yo"),
        (r"^wo+h*$", "wo"),
        (r"^huh+$", "ha"),
        (r"^duh+$", "da"),
        (r"^nah+$", "na"),
        (r"^bah+$", "ba"),
        (r"^lah+$", "la"),
        (r"^ohh+$", "o"),
        (r"^ooh+$", "u"),
    ]
]

LETTER_NAMES = {
    "en": {
        "a": "ei",
        "b": "bi",
        "c": "si",
        "d": "di",
        "e": "i",
        "f": "ef",
        "g": "ji",
        "h": "eich",
        "i": "ai",
        "j": "jei",
        "k": "kei",
        "l": "el",
        "m": "em",
        "n": "en",
        "o": "o",
        "p": "pi",
        "q": "kyu",
        "r": "ar",
        "s": "es",
        "t": "ti",
        "u": "yu",
        "v": "vi",
        "w": "dabalyu",
        "x": "eks",
        "y": "wai",
        "z": "zi",
    },
    "en-orthographic": {
        "a": "ay",
        "b": "bee",
        "c": "see",
        "d": "dee",
        "e": "ee",
        "f": "ef",
        "g": "jee",
        "h": "aych",
        "i": "eye",
        "j": "jay",
        "k": "kay",
        "l": "el",
        "m": "em",
        "n": "en",
        "o": "o",
        "p": "pee",
        "q": "kyu",
        "r": "ar",
        "s": "es",
        "t": "tee",
        "u": "yu",
        "v": "vee",
        "w": "doubleyu",
        "x": "ex",
        "y": "wy",
        "z": "zee",
    },
    "de": {
        "a": "a",
        "b": "be",
        "c": "tse",
        "d": "de",
        "e": "e",
        "f": "ef",
        "g": "ge",
        "h": "ha",
        "i": "i",
        "j": "yot",
        "k": "ka",
        "l": "el",
        "m": "em",
        "n": "en",
        "o": "o",
        "p": "pe",
        "q": "ku",
        "r": "er",
        "s": "es",
        "t": "te",
        "u": "u",
        "v": "fau",
        "w": "ve",
        "x": "iks",
        "y": "ipsilon",
        "z": "tset",
    },
    "fr": {
        "a": "a",
        "b": "be",
        "c": "se",
        "d": "de",
        "e": "oe",
        "f": "ef",
        "g": "zhe",
        "h": "ash",
        "i": "i",
        "j": "zhi",
        "k": "ka",
        "l": "el",
        "m": "em",
        "n": "en",
        "o": "o",
        "p": "pe",
        "q": "kyu",
        "r": "er",
        "s": "es",
        "t": "te",
        "u": "yu",
        "v": "ve",
        "w": "dublve",
        "x": "iks",
        "y": "igrek",
        "z": "zed",
    },
}
ACRONYMS = frozenset(
    """OK TV DJ CD DVD USA UK NYC MC VIP ID AM PM BFF DND SOS MVP CEO UFO ATM GPS LOL FBI CIA NBA NFL
BMW DNA EP LP OMG IDK TBH BRB RIP VHS ABC MTV BBC CNN XO AI UFC WWE NYPD LAPD DC AC PC TBD ETA PS IOU SUV RV TNT UV
MDMA LSD THC CBD PTSD ADHD OCD IQ EDM RNB BBQ DIY FYI GTA KFC NSA OG AKA VIP CDC IRS DMV LGBT LGBTQ NRA UN EU US LA
TLC AWOL MIA POV ESPN HBO USB SMS DM DMs IG FM AMA""".split()
)
SYMBOL_WORDS = {
    "&": {"en": "and", "de": "und", "fr": "et"},
    "%": {"en": "percent", "de": "prozent", "fr": "pour cent"},
    "$": {"en": "dollars", "de": "dollar", "fr": "dollars"},
    "+": {"en": "plus", "de": "plus", "fr": "plus"},
    "@": {"en": "at", "de": "at", "fr": "arobase"},
    "€": {"en": "euros", "de": "euro", "fr": "euros"},
    "£": {"en": "pounds", "de": "pfund", "fr": "livres"},
}
CURRENCY = frozenset("$€£")
ORDINAL_SUFFIXES = {
    "en": {"st", "nd", "rd", "th"},
    "fr": {"er", "re", "e", "ème", "eme"},
    "de": set(),
}
_APOSTROPHES = "'’ʼ"


def _is_word(text: str) -> bool:
    return bool(text) and all(c.isalpha() or c in _APOSTROPHES for c in text)


def collapse_repeats(word: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def vocable_label(word: str) -> str | None:
    folded = collapse_repeats(word.lower().translate(str.maketrans("’ʼ", "''")))
    for pattern, label in VOCABLES:
        if pattern.match(folded):
            return label
    return None


def is_acronym(word: str) -> bool:
    letters = word.replace("’", "'").strip("'")
    if not (
        2 <= len(letters) <= 4 and letters.isalpha() and letters.isascii() and letters.isupper()
    ):
        return False
    return not any(c in "AEIOU" for c in letters) or letters in ACRONYMS


# ---------------------------------------------------------------------------------------------
# G2P back ends (lazy, cached)

# espeak-ng releases accepted for v2 labels. The v2 manifest was built with 1.52.0 (summary.json
# "espeak_ng"); Ubuntu 24.04's 1.51 (the analysis image's apt package) gave identical output on all 187
# lines of fixtures/normalizer_byte_identity_v2.json (2026-10-18). Rule files change between releases,
# so any other version raises unless FA_KARA_ESPEAK_ANY_VERSION=1 (drift checks).
ESPEAK_ACCEPTED = ("1.51", "1.52")


@functools.lru_cache(maxsize=None)
def _espeak_backend(language: str):
    try:
        from phonemizer.backend import EspeakBackend
    except ImportError as error:  # pragma: no cover - documented dependency
        raise RuntimeError(
            "label rules v2 pronunciation labels need the phonemizer package and espeak-ng"
        ) from error
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    import os

    found = espeak_version()
    if (
        not found.startswith(ESPEAK_ACCEPTED)
        and os.environ.get("FA_KARA_ESPEAK_ANY_VERSION") != "1"
    ):
        raise RuntimeError(
            f"label rules v2 accept espeak-ng {' / '.join(ESPEAK_ACCEPTED)}, this is {found}: "
            "install that release (or set FA_KARA_ESPEAK_ANY_VERSION=1 knowingly)"
        )
    return EspeakBackend(
        ESPEAK_VOICES[language],
        with_stress=False,
        language_switch="keep-flags",
        words_mismatch="ignore",
        logger=logging.getLogger("phonemizer"),
    )


def espeak_version() -> str:
    try:
        from phonemizer.backend import EspeakBackend

        return ".".join(map(str, EspeakBackend.version()))
    except Exception:  # noqa: BLE001
        return "unavailable"


@functools.lru_cache(maxsize=None)
def _cmudict():
    try:
        from nltk.corpus import cmudict

        return cmudict.dict()
    except LookupError as error:  # pragma: no cover
        raise RuntimeError("CMUdict is not available to nltk") from error


@functools.lru_cache(maxsize=None)
def _pyphen(language: str):
    import pyphen

    return pyphen.Pyphen(lang=PYPHEN_LANGUAGES[language])


_SWITCH = re.compile(r"\((\w+)\)")


@functools.lru_cache(maxsize=65536)
def espeak_tokens(word: str, language: str) -> tuple[tuple[tuple[str, bool], ...], bool]:
    """(phone tokens, switched) for one word; ``switched`` when espeak used another language."""
    ipa = _espeak_backend(language).phonemize([word], strip=True, njobs=1)[0]
    switched = bool(_SWITCH.search(ipa))
    ipa = _SWITCH.sub("", ipa)
    return tuple(ipa_tokens(ipa, language)), switched


# French elided clitics (sung without their vowel): espeak reads "qu'tu" or "j't'ai" as foreign
# words, so they are peeled off and labelled by sound before the rest goes to espeak.
FRENCH_CLITICS = {
    "qu": (("k", False),),
    "j": (("zh", False),),
    "t": (("t", False),),
    "c": (("s", False),),
    "d": (("d", False),),
    "l": (("l", False),),
    "m": (("m", False),),
    "n": (("n", False),),
    "s": (("s", False),),
    "p": (("p", False),),
}
_CLITIC = re.compile(r"^(qu|[jtcdlmnsp])'(?=\w)")


@functools.lru_cache(maxsize=65536)
def word_pronunciation(word: str, language: str) -> tuple[tuple[tuple[str, bool], ...], str]:
    """Phone tokens of one word and where they came from: lexicon, cmudict, espeak, espeak-switch."""
    lowered = word.lower().translate(str.maketrans("’ʼ", "''"))
    vocable = vocable_label(word)
    if vocable:
        return tuple(
            (vocable, True),
        ), "lexicon"
    if language == "fr":
        match = _CLITIC.match(lowered)
        if match and len(lowered) > match.end():
            rest, source = word_pronunciation(lowered[match.end() :], language)
            return FRENCH_CLITICS[match.group(1)] + rest, source
    if language == "en":
        cmu = _cmudict()
        for candidate in (
            lowered,
            lowered.strip("'"),
            lowered.rstrip("'") + "g" if lowered.endswith("in'") else None,
        ):
            if candidate and candidate in cmu:
                return tuple(arpabet_tokens(cmu[candidate][0])), "cmudict"
    tokens, switched = espeak_tokens(word.translate(str.maketrans("’ʼ", "''")), language)
    return tokens, "espeak-switch" if switched else "espeak"


def pronunciation_items(word: str, language: str, item_type: int = 1) -> tuple[list[dict], str]:
    """Items for one Latin word under pronunciation labels; the origs concatenate to ``word``.

    Syllable items only when pyphen's surface syllables and the phone syllables agree in count.
    """
    tokens, source = word_pronunciation(word, language)
    pron = "".join(label for label, _ in tokens)
    if not pron:
        return [{"orig": word, "type": 0}], source
    phone_syllables = syllabify(list(tokens))
    surface = _pyphen(language).inserted(word).split("-") if "-" not in word else [word]
    surface = [s for s in surface if s]
    if (
        len(surface) > 1
        and len(surface) == len(phone_syllables)
        and "".join(surface) == word
        and all(phone_syllables)
    ):
        return [
            {"orig": s, "type": item_type, "pron": p} for s, p in zip(surface, phone_syllables)
        ], source
    return [{"orig": word, "type": item_type, "pron": pron}], source


# ---------------------------------------------------------------------------------------------
# Numbers, acronyms, symbols


def _number_words(digits: str, language: str, ordinal: bool = False) -> str:
    """Words of one digit string in the hint's language (English when the language has none)."""
    language = language if language in PRONUNCIATION_LANGUAGES else "en"
    value = int(digits)
    try:
        from num2words import num2words
    except ImportError as error:  # pragma: no cover - documented dependency
        raise RuntimeError("label rules v2 need the num2words package") from error
    leading = len(digits) - len(digits.lstrip("0")) if value else len(digits) - 1
    words = []
    if leading:
        zero = {"en": "o", "de": "null", "fr": "zéro"}[language]
        words += [zero] * leading
    if ordinal:
        words.append(num2words(value, lang=language, to="ordinal"))
    elif not leading and len(digits) == 4 and 1100 <= value <= 2099 and value % 1000 != 0:
        words.append(num2words(value, lang=language, to="year"))
    else:
        words.append(num2words(value, lang=language))
    return " ".join(words)


def words_to_items(
    orig: str,
    words: str,
    language: str,
    rules: LabelRules,
    item_type: int = 4,
    pronounce: bool | None = None,
) -> list[dict]:
    """One item for ``orig`` whose label reads ``words`` (pronounced or spelled per the rules)."""
    parts = [p for p in re.split(r"[\s\-]+", words.replace("-", " ")) if p]
    if pronounce is None:
        pronounce = language in rules.pronunciation
    if pronounce and rules.v3:
        pron = "".join(_v3_word_label(p, language, rules.sung_vowels) for p in parts)
    elif pronounce:
        pron = "".join(
            "".join(label for label, _ in word_pronunciation(p, language)[0]) for p in parts
        )
    else:
        pron = "".join(_spell(p) for p in parts)
    return (
        [{"orig": orig, "type": item_type, "pron": pron}] if pron else [{"orig": orig, "type": 0}]
    )


def _spell(word: str) -> str:
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z]", "", folded)


def number_items(
    digits: str,
    language: str,
    rules: LabelRules,
    ordinal: bool = False,
    currency: str | None = None,
) -> list[dict]:
    clean = re.sub(r"[^0-9]", "", digits)
    if not clean:
        return [{"orig": digits, "type": 0}]
    pronounce = (
        language in rules.pronunciation
    )  # under auto the number words are spelled like the words around
    language = language if language in PRONUNCIATION_LANGUAGES else "en"
    words = _number_words(clean, language, ordinal)
    if currency:
        words = f"{words} {SYMBOL_WORDS[currency][language]}"
    return words_to_items(digits, words, language, rules, pronounce=pronounce)


def acronym_items(word: str, language: str, rules: LabelRules) -> list[dict]:
    pronounce = language in rules.pronunciation
    language = language if language in PRONUNCIATION_LANGUAGES else "en"
    names = LETTER_NAMES[language if pronounce or language != "en" else "en-orthographic"]
    letters = word.replace("’", "'").strip("'").lower()
    return [{"orig": word, "type": 1, "pron": "".join(names[c] for c in letters)}]


def symbol_items(char: str, language: str, rules: LabelRules) -> list[dict]:
    pronounce = language in rules.pronunciation
    language = language if language in PRONUNCIATION_LANGUAGES else "en"
    words = SYMBOL_WORDS.get(char, {}).get(language)
    return (
        words_to_items(char, words, language, rules, item_type=0, pronounce=pronounce)
        if words
        else [{"orig": char, "type": 0}]
    )


def merge_number_runs(runs: list[list[str]]) -> list[list[str]]:
    """``10,000`` arrives as number / other / number runs; join thousands groups into one number run."""
    merged: list[list[str]] = []
    for kind, text in runs:
        if (
            kind == "number"
            and len(text) == 3
            and len(merged) >= 2
            and merged[-1] == ["other", ","]
            and merged[-2][0] == "number"
        ):
            merged.pop()
            merged[-1][1] += "," + text
            continue
        merged.append([kind, text])
    return merged


def latin_items(
    text: str, language: str, rules: LabelRules, spell, context: tuple | None = None
) -> list[dict]:
    """Items for one Latin run under v2/v3: vocable lexicon, acronyms, then pronunciation or spelling.

    ``spell(text)`` is the caller's v1 spelling path (uroman + pyphen literal syllables). ``context`` is
    (previous word, next word) of the line, lower-cased (v3 homographs only).
    """
    if not _is_word(text):
        return spell(text)
    vocable = vocable_label(text)
    if vocable:
        return [{"orig": text, "type": 1, "pron": vocable}]
    if is_acronym(text):
        return acronym_items(text, language, rules)
    if language in rules.pronunciation:
        if language == "en" and not (text.translate(_APOS_MAP) if rules.v3 else text).isascii():
            # A non-ASCII word in an English line is another language (não, tá, é): espeak-en would
            # read its letters by name (é → "e acute"); spell it as v1 does.
            return spell(text)
        if rules.v3:
            return pronunciation_items_v3(text, language, spell, context, sung=rules.sung_vowels)[0]
        return pronunciation_items(text, language)[0]
    if (
        rules.pronunciation
        and language in NON_LATIN_SCRIPT_LANGUAGES
        and text.isascii()
        and os.environ.get("FA_KARA_LATIN_AS_ENGLISH") == "1"
        and _english_known(text.lower())
    ):
        # Inference experiment (opt-in, 2026-10-19): English words in a Korean/Japanese/... song read
        # by English sound, as a v2 model learned English on English songs ("how's your day" was
        # spelled hows/your/day and sat 1-2 s late in Boy With Luv). Off: v2 output is unchanged.
        return pronunciation_items(text, "en")[0]
    return spell(text)


# Song languages whose own script is not Latin: a Latin word in their lyrics is a foreign (mostly
# English) word, not a romanisation of the song's language.
NON_LATIN_SCRIPT_LANGUAGES = frozenset(
    {
        "ko",
        "ja",
        "zh",
        "yue",
        "th",
        "ar",
        "fa",
        "ur",
        "he",
        "hi",
        "bn",
        "ta",
        "te",
        "mr",
        "gu",
        "kn",
        "ml",
        "pa",
        "ru",
        "uk",
        "bg",
        "sr",
        "mk",
        "be",
        "kk",
        "el",
        "ka",
        "hy",
        "km",
        "lo",
        "my",
    }
)


# ---------------------------------------------------------------------------------------------
# Label rules v3 (data/finetune-curation/LABEL_RULES_V3.md). Nothing below changes v1 or v2 output.

_APOS_MAP = str.maketrans("’ʼ", "''")


def _arpabet(phones: str) -> tuple[tuple[str, bool], ...]:
    return tuple(arpabet_tokens(phones.split()))


# Sung forms of casual / eye-dialect spellings in English lyrics, as ARPAbet (so syllables split like
# CMUdict words). Matched on the lower-cased word, curly apostrophes folded. Entries whose v2 label was
# already right are listed too, so the lexicon is the one place these words are defined.
SLANG_EN = {
    word: phones
    for words, phones in [
        ("tryna trynna", "T R AY1 N AH0"),
        ("gonna", "G AA1 N AH0"),
        ("gon' gon", "G AA1 N"),
        ("wanna", "W AA1 N AH0"),
        ("finna", "F IH1 N AH0"),
        ("gotta", "G AA1 T AH0"),
        ("lemme", "L EH1 M IY0"),
        ("gimme", "G IH1 M IY0"),
        ("kinda", "K AY1 N D AH0"),
        ("sorta", "S AO1 R T AH0"),
        ("outta", "AW1 T AH0"),
        ("imma i'ma i'mma ima", "AY1 M AH0"),
        ("y'all ya'll yall", "Y AO1 L"),
        ("ain't aint", "EY1 N T"),
        ("'cause cuz 'cuz coz 'cos", "K AH1 Z"),
        ("tho tho'", "DH OW1"),
        ("ya yuh", "Y AH0"),
        ("lil lil'", "L IH1 L"),
        ("'em", "AH0 M"),
        ("'bout bout", "B AW1 T"),
        ("boutta", "B AW1 T AH0"),
        ("'til til", "T IH1 L"),
        ("'round", "R AW1 N D"),
        ("'fore", "F AO1 R"),
        ("shoulda", "SH UH1 D AH0"),
        ("coulda", "K UH1 D AH0"),
        ("woulda", "W UH1 D AH0"),
        ("musta", "M AH1 S T AH0"),
        ("oughta", "AO1 T AH0"),
        ("hafta", "HH AE1 F T AH0"),
        ("gotcha", "G AA1 CH AH0"),
        ("betcha", "B EH1 CH AH0"),
        ("whatcha watcha", "W AH1 CH AH0"),
        ("gotchu", "G AA1 CH UW0"),
        ("lotta", "L AA1 T AH0"),
        ("dunno", "D AH0 N OW1"),
        ("c'mon cmon", "K AH0 M AA1 N"),
        ("shawty", "SH AO1 T IY0"),
        ("gyal", "G Y AA1 L"),
        ("innit", "IH1 N IH0 T"),
        ("thang", "TH AE1 NG"),
        ("ol'", "OW1 L"),
        ("'nough nuff", "N AH1 F"),
        ("wassup whassup", "W AH0 S AH1 P"),
        ("hunnid hunnit", "HH AH1 N IH0 D"),
    ]
    for word in words.split()
}

# Homographs: CMUdict's first reading is not the one lyrics mostly sing. (default, [(side, words, reading)]):
# the first rule whose neighbouring word (previous or next word of the line) is in ``words`` wins.
# Counts behind each choice (2,000 English v2 training songs) are in LABEL_RULES_V3.md.
_SUBJECTS = {
    "he",
    "she",
    "it",
    "who",
    "that",
    "which",
    "this",
    "everybody",
    "nobody",
    "somebody",
    "someone",
    "everyone",
    "love",
    "god",
    "legend",
    "spirit",
    "memory",
    "time",
}
HOMOGRAPHS_EN = {
    "live": (
        "L IH1 V",
        [
            (
                "next",
                {
                    "show",
                    "shows",
                    "music",
                    "band",
                    "stream",
                    "streaming",
                    "performance",
                    "concert",
                    "wire",
                    "broadcast",
                    "album",
                    "audience",
                    "footage",
                    "version",
                },
                "L AY1 V",
            )
        ],
    ),
    "lives": ("L AY1 V Z", [("prev", _SUBJECTS, "L IH1 V Z")]),
    "lived": ("L IH1 V D", []),
    "wind": ("W IH1 N D", [("next", {"up", "down"}, "W AY1 N D")]),
    "tear": (
        "T EH1 R",
        [
            (
                "prev",
                {
                    "a",
                    "single",
                    "one",
                    "every",
                    "no",
                    "another",
                    "last",
                    "first",
                    "lonely",
                    "little",
                    "lil",
                    "lil'",
                },
                "T IH1 R",
            ),
            (
                "next",
                {
                    "drop",
                    "drops",
                    "stained",
                    "falls",
                    "fell",
                    "fall",
                    "rolls",
                    "rolled",
                    "rolling",
                    "runs",
                    "ran",
                    "running",
                    "streams",
                    "streaming",
                    "gas",
                    "ducts",
                },
                "T IH1 R",
            ),
        ],
    ),
    "tears": (
        "T IH1 R Z",
        [
            ("prev", _SUBJECTS, "T EH1 R Z"),
            ("next", {"me", "him", "her", "us", "them", "apart", "open", "at"}, "T EH1 R Z"),
        ],
    ),
    "lead": (
        "L IY1 D",
        [
            (
                "prev",
                {
                    "of",
                    "with",
                    "like",
                    "full",
                    "pencil",
                    "heavy",
                    "hot",
                    "red",
                    "some",
                    "made",
                    "filled",
                    "pumped",
                },
                "L EH1 D",
            ),
            (
                "next",
                {
                    "pipe",
                    "pipes",
                    "paint",
                    "poisoning",
                    "balloon",
                    "zeppelin",
                    "belly",
                    "bullet",
                    "bullets",
                },
                "L EH1 D",
            ),
        ],
    ),
    "close": (
        "K L OW1 S",
        [
            (
                "next",
                {
                    "my",
                    "your",
                    "his",
                    "her",
                    "our",
                    "their",
                    "the",
                    "its",
                    "it",
                    "them",
                    "these",
                    "those",
                    "this",
                    "that",
                    "down",
                    "off",
                    "out",
                    "shop",
                    "ranks",
                    "door",
                    "doors",
                    "eyes",
                },
                "K L OW1 Z",
            )
        ],
    ),
    "read": (
        "R IY1 D",
        [
            (
                "prev",
                {
                    "have",
                    "has",
                    "had",
                    "i've",
                    "you've",
                    "we've",
                    "they've",
                    "she's",
                    "he's",
                    "been",
                    "was",
                    "were",
                    "already",
                    "once",
                },
                "R EH1 D",
            )
        ],
    ),
    "bow": (
        "B AW1",
        [
            (
                "prev",
                {
                    "my",
                    "your",
                    "her",
                    "his",
                    "pink",
                    "red",
                    "favorite",
                    "favourite",
                    "rain",
                    "silver",
                    "golden",
                },
                "B OW1",
            ),
            ("next", {"tie", "ties", "arrow", "arrows", "string", "legged"}, "B OW1"),
        ],
    ),
    "wound": (
        "W UW1 N D",
        [("next", {"up", "down", "around", "round", "tight", "tighter", "back"}, "W AW1 N D")],
    ),
    "use": (
        "Y UW1 Z",
        [
            (
                "prev",
                {"no", "the", "what's", "whats", "any", "much", "of", "a", "there's"},
                "Y UW1 S",
            )
        ],
    ),
    "used": ("Y UW1 Z D", [("next", {"to", "ta"}, "Y UW1 S T")]),
    "bass": ("B EY1 S", []),
    "desert": (
        "D EH1 Z ER0 T",
        [
            (
                "prev",
                {
                    "won't",
                    "never",
                    "not",
                    "wouldn't",
                    "don't",
                    "didn't",
                    "will",
                    "would",
                    "to",
                    "i'll",
                    "can't",
                    "gonna",
                },
                "D IH0 Z ER1 T",
            )
        ],
    ),
    "excuse": (
        "IH0 K S K Y UW1 S",
        [
            (
                "next",
                {"me", "my", "us", "him", "her", "them", "it", "yourself", "myself", "you"},
                "IH0 K S K Y UW1 Z",
            )
        ],
    ),
    "record": (
        "R EH1 K ER0 D",
        [
            (
                "prev",
                {
                    "to",
                    "let's",
                    "gonna",
                    "wanna",
                    "will",
                    "i'll",
                    "we'll",
                    "can",
                    "can't",
                    "don't",
                    "could",
                },
                "R IH0 K AO1 R D",
            )
        ],
    ),
    "reading": ("R IY1 D IH0 NG", []),
    "putting": ("P UH1 T IH0 NG", []),
    "perfect": ("P ER1 F IH0 K T", []),
    "sow": ("S OW1", []),
    "dove": (
        "D AH1 V",
        [
            ("prev", {"a", "the", "white", "little", "my", "like"}, "D AH1 V"),
            ("next", {"into", "in", "deep", "headfirst", "off", "under"}, "D OW1 V"),
        ],
    ),
    "dr": ("D AA1 K T ER0", []),
}

# Dropped-g spellings CMUdict lists with another reading (a name or a reduced form): read as -ing too.
DROPPED_G_IN_CMU = frozenset("nothin goin lovin killin talkin makin rollin lookin rockin".split())

# Single words espeak reads out of context: French "y" (il y a) as the letter name igrek, German "'n"
# (ein) as the letter name en.
FRENCH_WORDS_V3 = {"y": (("i", True),)}
GERMAN_WORDS_V3 = {"'n": (("n", False),)}

# English contractions in German/French lines (espeak-de read "we're" as vere, "won't" as vont).
_EN_CONTRACTION = re.compile(
    r"^(i|you|we|they|he|she|it|that|what|there|let|who|where|how)'(m|re|ve|ll|d|s)$"
    r"|^[a-z]+n't$|^'(cause|cuz|bout|til|em)$"
)


def _english_contraction(lowered: str) -> bool:
    """An English contraction or dropped-g form (German mein' / dein' are not: no "meing")."""
    if lowered.endswith("in'"):
        return lowered[:-1] + "g" in _cmudict()
    return bool(_EN_CONTRACTION.match(lowered))


def _english_known(lowered: str) -> bool:
    """Whether a lower-cased word is English the English path knows (CMUdict, lexicon, dropped g)."""
    cmu = _cmudict()
    base = lowered.rstrip("'")
    return (
        lowered in cmu
        or lowered.strip("'") in cmu
        or lowered in SLANG_EN
        or (base.endswith("in") and base + "g" in cmu)
    )


def _homograph(lowered: str, context: tuple | None) -> str:
    default, rules = HOMOGRAPHS_EN[lowered]
    previous, following = context or ("#", "#")
    for side, words, reading in rules:
        if (previous if side == "prev" else following) in words:
            return reading
    return default


def _english_lexicons(lowered: str, context: tuple | None, sung: bool = False):
    """(tokens, source) from the v3 English lexicons and CMUdict, or None when espeak must read it.

    ``sung`` (v3-sung): reduced vowels take their spelled vowel and CMUdict's reading is the one that
    aligns best with the spelling (``sung_vowels``); otherwise CMUdict's first reading as written."""

    def tokens_of(readings):
        if sung:
            return sung_vowels([arpabet_units(phones) for phones in readings], lowered)
        return tuple(arpabet_tokens(readings[0]))

    if lowered in SLANG_EN:
        return tokens_of([SLANG_EN[lowered].split()]), "slang"
    if lowered in HOMOGRAPHS_EN:
        return tokens_of([_homograph(lowered, context).split()]), "homograph"
    cmu = _cmudict()
    base = lowered.rstrip("'")
    if (
        len(base) >= 4
        and base.endswith("in")
        and base + "g" in cmu
        and (lowered.endswith("in'") or base not in cmu or base in DROPPED_G_IN_CMU)
    ):
        readings = []
        for reading in cmu[base + "g"] if sung else cmu[base + "g"][:1]:
            phones = list(reading)
            if phones[-1] == "NG":
                phones[-1] = "N"  # sung as written: nothin' → nathin, gettin → getin
            readings.append(phones)
        return tokens_of(readings), "dropped_g"
    for candidate in (lowered, lowered.strip("'")):
        if candidate in cmu:
            if _cmu_spellout(candidate, tuple(arpabet_tokens(cmu[candidate][0]))):
                return (), "letter_spellout"
            return tokens_of(cmu[candidate]), "cmudict"
    return None


# ---------------------------------------------------------------------------------------------
# v3-sung: sung unstressed vowels. A reduced vowel (CMUdict AH0, espeak ə / ᵻ) is a slot ``@a`` /
# ``@i`` (the v3 label after the @) until the phones are aligned to the word's letters; a slot over
# exactly one vowel letter takes that letter (y reads i), any other slot keeps its v3 label.


def arpabet_units(phones) -> list[tuple[str, bool]]:
    """arpabet_tokens with AH0 kept apart (as the slot ``@a``) before the stress digit is dropped."""
    units = []
    for phone in phones:
        if phone == "AH0":
            units.append(("@a", True))
        else:
            units.extend(arpabet_tokens([phone]))
    return units


# Spellings of a consonant label besides the label itself (as cheap as the label's own letters).
_GRAPHEMES = {
    "k": {"c", "ck", "ch", "q", "qu", "cc", "kh", "x", "lk"},
    "g": {"gg", "gh", "gu", "x"},
    "f": {"ph", "gh", "ff"},
    "s": {"c", "ss", "sc", "ps", "st", "x", "z"},
    "z": {"s", "zz", "x", "ss"},
    "sh": {"ti", "ci", "si", "ss", "ch", "sch", "s", "sci", "ce"},
    "zh": {"s", "si", "g", "z", "ge"},
    "ch": {"tch", "t", "ti", "c", "cz"},
    "j": {"g", "dg", "dge", "gg", "ge", "gi", "d", "di"},
    "dh": {"th"},
    "th": {"th"},
    "ng": {"n", "ngue"},
    "t": {"tt", "ed", "th", "pt", "bt", "d"},
    "d": {"dd", "ed", "ld", "t", "tt"},
    "n": {"nn", "kn", "gn", "pn", "mn"},
    "m": {"mm", "mb", "mn", "lm"},
    "r": {"rr", "wr", "rh"},
    "l": {"ll"},
    "p": {"pp"},
    "b": {"bb"},
    "v": {"vv", "f", "ph"},
    "w": {"wh", "u", "o"},
    "y": {"i", "j", "e", "u"},
    "h": {"wh"},
}
_VOWEL_LETTERS = frozenset("aeiouy")
_SPELLED_VOWEL = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "y": "i"}


def _unit_cost(label: str, letters: str) -> float:
    """Cost of one label unit covering ``letters``. Each vowel costs the same whichever vowel letters it
    covers and each consonant nothing over any of its spellings, so readings compete on structure (number
    and place of vowels, silent letters, letterless phones) and a tie goes to CMUdict's order, never to
    the reading that looks most like the spelling."""
    vowel = label.startswith("@") or label in (
        "a",
        "e",
        "i",
        "o",
        "u",
        "ai",
        "au",
        "ei",
        "oi",
        "er",
    )
    if (
        not letters
    ):  # a reduced vowel with no vowel letter (rhythm), the u of "use", the 2nd phone of x
        return 0.8 if label.startswith("@") else 0.3 if label in ("y", "w", "s", "z") else 1.5
    if vowel:
        core = (
            letters if label == "er" or label.startswith("@") else letters.rstrip("wyrh") or letters
        )  # ay, ow; y alone
        allowed = _VOWEL_LETTERS | ({"r"} if label == "er" else set())
        return 0.2 if core and set(core) <= allowed and len(letters) <= 4 else 9.0
    return 0.0 if letters == label or letters in _GRAPHEMES.get(label, ()) else 9.0


def align_spelling(units, word: str) -> tuple[float, list[str], list[int]]:
    """(cost, letters each unit covers, index of each unit's first letter) of the cheapest monotonic
    alignment of label units to the word's a-z letters; letters no unit covers are silent (e, h, w, u, gh
    cost 0.5, others 1.0). A unit covering no letter gets the index where it sits."""
    letters = re.sub(r"[^a-z]", "", word.lower())
    n, m = len(letters), len(units)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            here = cost[i][j]
            if here == inf:
                continue
            if i < n:  # silent letter
                step = here + (0.5 if letters[i] in "ehwu" else 1.0)
                if step < cost[i + 1][j]:
                    cost[i + 1][j], back[i + 1][j] = step, (i, j)
            if j < m:
                for k in range(0, min(4, n - i) + 1):
                    step = here + _unit_cost(units[j][0], letters[i : i + k])
                    if step < cost[i + k][j + 1]:
                        cost[i + k][j + 1], back[i + k][j + 1] = step, (i, j)
    spans, starts = [""] * m, [0] * m
    i, j = n, m
    while (i, j) != (0, 0) and back[i][j] is not None:
        pi, pj = back[i][j]
        if pj == j - 1:
            spans[pj], starts[pj] = letters[pi:i], pi
        i, j = pi, pj
    return cost[n][m], spans, starts


# A later CMUdict reading replaces the first only when it aligns with the spelling better by more than
# this (above one silent e, below one silent consonant): are stays ar (not ER0), believe stays bliv.
SUNG_READING_MARGIN = 0.6


def sung_vowels(readings, word: str) -> tuple[tuple[str, bool], ...]:
    """The first reading (unit lists with reduced slots), or a later one that aligns with ``word``'s letters
    better by SUNG_READING_MARGIN; each slot over one vowel letter reads as that vowel, any other as v3."""
    best = None
    for units in readings:
        score, spans, _ = align_spelling(units, word)
        if best is None or score < best[0] - SUNG_READING_MARGIN:
            best = (score, units, spans)
    _, units, spans = best
    tokens = []
    for (label, vowel), span in zip(units, spans):
        if label.startswith("@"):
            label = _SPELLED_VOWEL[span] if len(span) == 1 else label[1:]
        tokens.append((label, vowel))
    return tuple(tokens)


def _abbreviation(word: str) -> bool:
    """Lower-case acronyms keep their letter names: an acronym (DJ, tv), or 2-4 letters without a vowel or
    a doubled letter (km, dms, fr). Doubled letters are onomatopoeia (brr, psst, skrrt)."""
    if word.upper() in ACRONYMS:
        return True
    return (
        2 <= len(word) <= 4
        and not any(c in "aeiouy" for c in word)
        and not any(a == b for a, b in zip(word, word[1:]))
    )


def _cmu_spellout(word: str, tokens) -> bool:
    """A CMUdict acronym entry for a lower-case lyric word (mga → em ji ei)."""
    if not (len(word) >= 2 and word.isascii() and word.isalpha()) or _abbreviation(word):
        return False
    return "".join(label for label, _ in tokens) == "".join(LETTER_NAMES["en"][c] for c in word)


@functools.lru_cache(maxsize=262144)
def _espeak_v3(
    word: str, language: str, sung: bool = False
) -> tuple[tuple[tuple[str, bool], ...], str | None]:
    """(phone tokens, language espeak switched to or None) for one word or a spaced letter string;
    ``sung`` (v3-sung, English): reduced vowels take their spelled vowel (``sung_vowels``)."""
    ipa = _espeak_backend(language).phonemize([word], strip=True, njobs=1)[0]
    switch = _SWITCH.search(ipa)
    sung = sung and language == "en" and not switch
    tokens = ipa_tokens(_SWITCH.sub("", ipa), language, reduced=sung)
    return (sung_vowels([tokens], word) if sung else tuple(tokens)), (
        switch.group(1) if switch else None
    )


def _espeak_label(text: str, language: str) -> str:
    return "".join(label for label, _ in _espeak_v3(text, language)[0])


def _espeak_spellout(word: str, label: str, language: str) -> bool:
    """espeak read the word (or its leading consonants) letter by letter: skrrt → es kei ar…, gbe → ji bi i,
    nhlonipho → en eich lanifo. Compared with espeak's own letter names, so it holds for every voice."""
    if not (len(word) >= 2 and word.isascii() and word.isalpha()) or _abbreviation(word):
        return False
    names = [_espeak_label(c, language) for c in word]
    if not names[0] or not label.startswith(names[0]):
        return False
    if label == "".join(names) or label == _espeak_label(" ".join(word), language):
        return True
    # Spelled leading consonants: the letter names' vowels stand before another consonant (s|krrt,
    # n|hlonipho, g|bedu), so they cannot be the word's own vowel (German b|eef reads "be…" anyway).
    lead = 0
    while lead < len(word) and word[lead] not in "aeiouy":
        lead += 1
    for k in range(1, min(lead - 1, len(word) - 2) + 1):
        prefix = "".join(names[:k])
        if label.startswith(prefix) and label == prefix + _espeak_label(word[k:], language):
            return True
    return False


def _collapse_letters(label: str) -> str:
    return re.sub(r"(.)\1+", r"\1", label)


@functools.lru_cache(maxsize=262144)
def word_pronunciation_v3(
    word: str, language: str, context: tuple | None = None, sung: bool = False
):
    """(phone tokens, source, pronunciation language) of one word under v3 (``sung``: v3-sung).

    Sources: lexicon, slang, homograph, dropped_g, cmudict, espeak, espeak-switch, english:<source> (an
    English word in a de/fr line), letter_spellout (no tokens: the caller spells the word)."""
    lowered = word.lower().translate(_APOS_MAP)
    vocable = vocable_label(word)
    if vocable:
        return ((vocable, True),), "lexicon", language
    if language == "fr":
        if lowered in FRENCH_WORDS_V3:
            return FRENCH_WORDS_V3[lowered], "lexicon", language
        match = _CLITIC.match(lowered)
        if match and len(lowered) > match.end():
            rest, source, _ = word_pronunciation_v3(lowered[match.end() :], language, sung=sung)
            if source == "letter_spellout":
                return (), source, language
            return FRENCH_CLITICS[match.group(1)] + rest, source, language
    if language == "de" and lowered in GERMAN_WORDS_V3:
        return GERMAN_WORDS_V3[lowered], "lexicon", language
    if language in ("de", "fr") and _english_contraction(lowered) and _english_known(lowered):
        tokens, source, _ = word_pronunciation_v3(word, "en", context, sung)
        return tokens, "english:" + source, "en"
    if language == "en":
        found = _english_lexicons(lowered, context, sung)
        if found:
            return (*found, language)
    tokens, switch = _espeak_v3(word.translate(_APOS_MAP), language, sung)
    if language in ("de", "fr") and switch == "en" and _english_known(lowered):
        english, source, _ = word_pronunciation_v3(word, "en", context, sung)
        if english:
            return english, "english:" + source, "en"
    plain = (
        _espeak_v3(word.translate(_APOS_MAP), language)[0] if sung else tokens
    )  # spell-outs as in v3
    label = "".join(t for t, _ in plain)
    if _espeak_spellout(lowered, label, language):
        if language in ("de", "fr") and _english_known(lowered):  # "hi" in a German line
            english, source, _ = word_pronunciation_v3(word, "en", context, sung)
            if english:
                return english, "english:" + source, "en"
        return (), "letter_spellout", language
    return tokens, "espeak-switch" if switch else "espeak", language


def _v3_word_label(word: str, language: str, sung: bool = False) -> str:
    tokens, source, _ = word_pronunciation_v3(word, language, sung=sung)
    return _spell(word) if source == "letter_spellout" else "".join(label for label, _ in tokens)


def aligned_surface(tokens, word: str) -> list[str] | None:
    """v3-sung: the word's spelling cut where its phone syllables are cut, through the letters each
    phone was aligned to (intuition → in|tu|i|tion where pyphen gives in-tu-ition); silent letters stay
    with the syllable before. None when the alignment is impossible or a syllable gets no letter."""
    cuts = syllable_cuts(list(tokens))
    positions = [k for k, char in enumerate(word) if "a" <= char.lower() <= "z"]
    if not cuts or not positions:
        return None
    score, spans, starts = align_spelling(tokens, word)
    if score >= 9.0:
        return None
    bounds = []
    for cut in cuts:
        lettered = [starts[k] for k in range(cut, len(tokens)) if spans[k]]
        if not lettered:
            return None
        bounds.append(positions[lettered[0]])
    edges = [0, *bounds, len(word)]
    if any(a >= b for a, b in zip(edges, edges[1:])):
        return None
    return [word[a:b] for a, b in zip(edges, edges[1:])]


def pronunciation_items_v3(
    word: str, language: str, spell, context: tuple | None = None, sung: bool = False
) -> tuple[list[dict], str]:
    """v3 counterpart of pronunciation_items; a letter spell-out falls back to ``spell`` with repeated
    letters collapsed (skrrt → skrt)."""
    lowered = word.lower().translate(_APOS_MAP)
    tokens, source, used = word_pronunciation_v3(
        word, language, context if lowered in HOMOGRAPHS_EN else None, sung
    )
    if source == "letter_spellout":
        spelled = _collapse_letters("".join(item.get("pron", "") for item in spell(word)))
        return (
            [{"orig": word, "type": 5, "pron": spelled}] if spelled else [{"orig": word, "type": 0}]
        ), source
    pron = "".join(label for label, _ in tokens)
    if not pron:
        return [{"orig": word, "type": 0}], source
    phone_syllables = syllabify(list(tokens))
    surface = _pyphen(used).inserted(word).split("-") if "-" not in word else [word]
    surface = [s for s in surface if s]
    if (
        len(surface) > 1
        and len(surface) == len(phone_syllables)
        and "".join(surface) == word
        and all(phone_syllables)
    ):
        return [{"orig": s, "type": 1, "pron": p} for s, p in zip(surface, phone_syllables)], source
    if sung and len(phone_syllables) > 1 and "-" not in word and all(phone_syllables):
        # v3-sung: pyphen disagrees with the phones (in-tu-ition vs in·tu·i·shon); cut the spelling
        # through the phone-to-letter alignment instead of keeping one long item.
        aligned = aligned_surface(tokens, word)
        if aligned and len(aligned) == len(phone_syllables):
            return [
                {"orig": s, "type": 1, "pron": p} for s, p in zip(aligned, phone_syllables)
            ], source
    return [{"orig": word, "type": 1, "pron": pron}], source


# Per-line language (v3). fastText lid.176 (the compressed .ftz, CC BY-SA 3.0, the model the manifest
# builder's document-level LID uses); the file is pinned by hash so training and inference agree.
LID_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
LID_CANDIDATES = ("/opt/fa_kara_models/lid.176.ftz", "/models/lid.176.ftz")
_LID_WORD = re.compile(r"[^\W\d_]+(?:['’ʼ][^\W\d_]+)*")
LID_ALPHABETS = {
    "de": frozenset("abcdefghijklmnopqrstuvwxyzäöüß'"),
    "fr": frozenset("abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ'"),
}
# A German/French line inside an English song also needs two of that language's function words or one
# of its accented letters (fastText calls patois "dem … dem" lines German).
LID_EVIDENCE = {
    "de": (
        frozenset(
            "ich und nicht ist du der die das ein eine einen mein meine dich mich mir dir wir ihr "
            "auf mit für zu nur auch noch wie was wenn aber nein ja sie es".split()
        ),
        "äöüß",
    ),
    "fr": (
        frozenset(
            "je tu il elle nous vous le la les des du et est pas que qui mon ma mes ton ta tes un "
            "une dans pour avec moi toi suis c'est j'ai sur au aux ce".split()
        ),
        "éèêàçùâîôœ",
    ),
}


def lid_model_path() -> str:
    import os

    found = os.environ.get("FA_KARA_LID_MODEL") or next(
        (p for p in LID_CANDIDATES if os.path.exists(p)), None
    )
    if not found or not os.path.exists(found):
        raise RuntimeError(
            "label rules v3 need fastText lid.176.ftz: set FA_KARA_LID_MODEL "
            f"(sha256 {LID_SHA256}) or install it at {LID_CANDIDATES[0]}"
        )
    return found


@functools.lru_cache(maxsize=None)
def _lid_model():
    import hashlib
    import os

    path = lid_model_path()
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != LID_SHA256 and os.environ.get("FA_KARA_LID_ANY_MODEL") != "1":
        raise RuntimeError(
            f"label rules v3 use lid.176.ftz sha256 {LID_SHA256}; {path} is {digest}"
        )
    try:
        import fasttext
    except ImportError as error:  # pragma: no cover - documented dependency
        raise RuntimeError(
            "label rules v3 need the fasttext package (per-line language)"
        ) from error
    return fasttext.load_model(path)


def line_lid(text: str) -> tuple[str, float]:
    """(language, probability) of fastText's top prediction for lower-cased words."""
    score, label = _lid_model().f.predict(text + "\n", 1, 0.0, "strict")[0]
    return label.replace("__label__", ""), float(score)


@functools.lru_cache(maxsize=65536)
def line_language(line: str, language: str) -> str:
    """The hint one line of an en/de/fr song is normalised with under v3.

    Only the line's Latin words are identified. In an English song a line goes to another language when
    fastText is confident and most of its words are not English the English path knows (CMUdict share);
    German/French only with that language's letters and evidence (LID_EVIDENCE), everything else to the
    generic Latin path ``auto``. In a German/French song an English line (fastText ≥ 0.8, ≥ 75 % of ≥ 2 words in CMUdict)
    goes to English G2P. Lines of fewer than two Latin words keep the song's language."""
    if language not in PRONUNCIATION_LANGUAGES:
        return language
    words = [
        w.lower().translate(_APOS_MAP)
        for w in _LID_WORD.findall(line)
        if all("LATIN" in unicodedata.name(c, "") or c in _APOSTROPHES for c in w)
    ]
    if len(words) < 2:
        return language
    top, confidence = line_lid(" ".join(words))
    known = [_english_known(w) for w in words if not vocable_label(w)]
    share = sum(known) / len(known) if known else 1.0
    if language == "en":
        if top == "en" or not (
            (confidence >= 0.6 and share < 0.5) or (confidence >= 0.9 and share < 0.8)
        ):
            return language
        if top in LID_ALPHABETS and set("".join(words)) <= LID_ALPHABETS[top]:
            function_words, letters = LID_EVIDENCE[top]
            if sum(w in function_words for w in words) >= 2 or any(
                c in letters for c in "".join(words)
            ):
                return top
        return "auto"
    if top == "en" and confidence >= 0.8 and len(known) >= 2 and share >= 0.75:
        return "en"
    return language


def latin_run_languages(runs, language: str, rules: LabelRules) -> dict[int, str]:
    """v3: language overrides for single Latin words (runs) of one line part, keyed by run index.

    (a) A song whose script is not Latin (ko, ja, zh, ru, ...): when at least 75 % of the line's Latin words
        are English the English path knows, those ASCII words are labelled as English (by sound under v3),
        as English is labelled in English songs ("how's your day" in a Korean song).
    (b) A Latin line of an en/de/fr song (after ``line_language``): a run of >= 2 consecutive words in
        another language switches to that language's rules: English runs in a German/French line
        (all words English-known, fastText en >= 0.8), foreign runs in an English line (no word
        English-known, fastText non-en >= 0.8 -> de/fr when their alphabet fits, else the generic
        Latin path ``auto``). Single words never switch: short words are too ambiguous (die, bin, me).
    Whitespace/punctuation may bridge words; intervening lexical, numeric or symbol runs break groups.
    Orthographic rule sets have no pronunciation languages, so they get no overrides."""
    if not rules.v3 or not rules.pronunciation:
        return {}
    latin = [(i, text) for i, (kind, text) in enumerate(runs) if kind == "latin" and _is_word(text)]
    if not latin:
        return {}
    words = [(i, text, text.lower().translate(_APOS_MAP)) for i, text in latin]
    overrides: dict[int, str] = {}
    if language in NON_LATIN_SCRIPT_LANGUAGES:
        counted = [(i, t, w) for i, t, w in words if not vocable_label(w)]
        known = [w.isascii() and _english_known(w) for _, t, w in counted]
        if known and sum(known) / len(known) >= 0.75:
            overrides.update({i: "en" for (i, t, w), k in zip(counted, known) if k})
        return overrides
    if language not in PRONUNCIATION_LANGUAGES:
        return overrides

    def flush(group, english):
        if len(group) < 2 or english is None:
            return
        letters = "".join(w for _, w in group)
        top, confidence = line_lid(" ".join(w for _, w in group))
        if confidence < 0.8:
            return
        if english and language != "en" and top == "en":
            overrides.update({i: "en" for i, _ in group})
        elif not english and language == "en" and top != "en":
            target = (
                top
                if top in ("de", "fr") and set(letters) <= LID_ALPHABETS.get(top, set())
                else "auto"
            )
            overrides.update({i: target for i, _ in group})

    group, status = [], None
    previous = None
    for i, text, w in words:
        # Whitespace and punctuation are transparent; lexical, numeric and symbol runs are
        # barriers. Never join two foreign words across an intervening non-Latin word/number.
        if previous is not None and any(
            not all(c.isspace() or unicodedata.category(c).startswith("P") for c in gap)
            for _, gap in runs[previous + 1 : i]
        ):
            flush(group, status)
            group, status = [], None
        previous = i
        english = status if vocable_label(w) else (w.isascii() and _english_known(w))
        if english != status:
            flush(group, status)
            group = []
        group.append((i, w))
        status = english
    flush(group, status)
    return overrides
