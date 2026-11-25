"""Pronunciation-based romanization of Hangul for acoustic training labels.

The vendored normalizer romanizes each syllable block by spelling: 같이 →
gat+i, 국물 → gug+mul. Singers produce ga+chi and gung+mul. This applies the
standard pronunciation rules (표준 발음법) inside each run of adjacent Hangul
syllables, then romanizes each syllable by sound, so a consonant carried into
the next syllable is labelled in the time that syllable is sung: 먹어 → meo+geo.

Rules: ㅎ deletion and aspiration, palatalization, liaison (연음), coda
neutralization, nasalization, ㄴ/ㄹ assimilation and post-obstruent tensing.
Lexical exceptions (맛없다, 의견란, 읽고) are not modelled. Training labels
(data/finetune-curation) and inference (main.normalize_source_lines) share this
module, so the aligner hears the same letters it was trained on.
"""

from __future__ import annotations

VERSION = "ko_pronunciation_v1"
# Label rules v2 (DATASET_FIXES_2026-10-18.md; V7_PLAN.md "Korean ㅢ after a consonant"): 표준 발음법 제5항
# — ㅢ after a consonant onset is [ㅣ] (희망 → hi+mang, 무늬 → mu+ni), a non-initial 의 is [ㅣ] inside a
# run and the run-final particle 의 is [ㅔ] (나의 → na+e); a run-initial 의 stays ui; ㄺ before ㄱ is
# [ㄹ] + tensed ㄱ (읽고 → il+kko). v1 is unchanged.
VERSION_V2 = "ko_pronunciation_v2"

ONSETS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
VOWELS = [
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]
CODAS = [
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]

ONSET_ROMAN = dict(
    zip(
        ONSETS,
        [
            "g",
            "kk",
            "n",
            "d",
            "tt",
            "r",
            "m",
            "b",
            "pp",
            "s",
            "ss",
            "",
            "j",
            "jj",
            "ch",
            "k",
            "t",
            "p",
            "h",
        ],
    )
)
VOWEL_ROMAN = dict(
    zip(
        VOWELS,
        [
            "a",
            "ae",
            "ya",
            "yae",
            "eo",
            "e",
            "yeo",
            "ye",
            "o",
            "wa",
            "wae",
            "oe",
            "yo",
            "u",
            "wo",
            "we",
            "wi",
            "yu",
            "eu",
            "ui",
            "i",
        ],
    )
)
CODA_ROMAN = {"": "", "ㄱ": "k", "ㄴ": "n", "ㄷ": "t", "ㄹ": "l", "ㅁ": "m", "ㅂ": "p", "ㅇ": "ng"}

# Compound codas: (stays, moves to the next syllable before a vowel).
COMPOUND = {
    "ㄳ": ("ㄱ", "ㅅ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}
NEUTRAL = {
    **{c: "ㄱ" for c in "ㄱㄲㅋㄳㄺ"},
    **{c: "ㄴ" for c in "ㄴㄵㄶ"},
    **{c: "ㄷ" for c in "ㄷㅅㅆㅈㅊㅌㅎ"},
    **{c: "ㄹ" for c in "ㄹㄼㄽㄾㅀ"},
    **{c: "ㅁ" for c in "ㅁㄻ"},
    **{c: "ㅂ" for c in "ㅂㅍㅄㄿ"},
    "ㅇ": "ㅇ",
    "": "",
}
ASPIRATED = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
NASAL = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}


def is_syllable(char: str) -> bool:
    return len(char) == 1 and "가" <= char <= "힣"


def decompose(char: str) -> list[str]:
    code = ord(char) - 0xAC00
    return [ONSETS[code // 588], VOWELS[code % 588 // 28], CODAS[code % 28]]


def _join(cur: list[str], nxt: list[str], v2: bool = False) -> None:
    """Apply the boundary rules between two adjacent syllables in place."""
    if v2 and cur[2] == "ㄺ" and nxt[0] == "ㄱ":
        # 제11항 다만: ㄺ before ㄱ is [ㄹ] and the ㄱ tenses (읽고 → il+kko, 맑게 → mal+kke).
        cur[2], nxt[0] = "ㄹ", "ㄲ"
    coda, onset = cur[2], nxt[0]
    # ㅎ in the coda: aspirates ㄱㄷㅈ, tenses ㅅ, drops before vowels, nasal before ㄴ.
    if coda in ("ㅎ", "ㄶ", "ㅀ"):
        rest = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[coda]
        if onset in ("ㄱ", "ㄷ", "ㅈ"):
            nxt[0], cur[2] = ASPIRATED[onset], rest
        elif onset == "ㅅ":
            nxt[0], cur[2] = "ㅆ", rest
        elif onset == "ㅇ":
            cur[2] = rest
        elif onset == "ㄴ":
            cur[2] = rest or "ㄴ"
        coda, onset = cur[2], nxt[0]
    # Obstruent + ㅎ aspirates: 축하 → 추카, 닫히다 → 다치다, 앉히다 → 안치다.
    if onset == "ㅎ" and coda:
        stays, moved = COMPOUND.get(coda, ("", coda))
        base = NEUTRAL.get(moved, moved) if moved not in ("ㅈ",) else "ㅈ"
        if base in ASPIRATED:
            aspirated = ASPIRATED[base]
            if aspirated == "ㅌ" and nxt[1] == "ㅣ":
                aspirated = "ㅊ"
            cur[2], nxt[0] = stays, aspirated
            coda, onset = cur[2], nxt[0]
    # Palatalization: 같이 → 가치, 굳이 → 구지, 핥이다 → 할치다.
    if onset == "ㅇ" and nxt[1] == "ㅣ" and coda in ("ㄷ", "ㅌ", "ㄾ"):
        cur[2], nxt[0] = ("ㄹ", "ㅊ") if coda == "ㄾ" else ("", "ㅈ" if coda == "ㄷ" else "ㅊ")
        coda, onset = cur[2], nxt[0]
    # Liaison: a coda before a vowel is sung as the next onset; 없어 → 업써.
    if onset == "ㅇ" and coda not in ("", "ㅇ"):
        stays, moved = COMPOUND.get(coda, ("", coda))
        if moved == "ㅅ" and stays in ("ㄱ", "ㅂ"):
            moved = "ㅆ"
        cur[2], nxt[0] = stays, moved
        coda, onset = cur[2], nxt[0]
    coda = NEUTRAL[coda]
    # Nasalization and ㄹ assimilation.
    if coda in NASAL and onset in ("ㄴ", "ㅁ"):
        coda = NASAL[coda]
    elif coda in ("ㅁ", "ㅇ") and onset == "ㄹ":
        onset = "ㄴ"
    elif coda in ("ㄱ", "ㅂ") and onset == "ㄹ":
        coda, onset = NASAL[coda], "ㄴ"
    elif coda == "ㄴ" and onset == "ㄹ":
        coda = "ㄹ"
    elif coda == "ㄹ" and onset == "ㄴ":
        onset = "ㄹ"
    # Tensing after an unreleased obstruent: 학교 → 학꾜.
    if coda in ("ㄱ", "ㄷ", "ㅂ") and onset in TENSE:
        onset = TENSE[onset]
    cur[2], nxt[0] = coda, onset


def romanize_run(text: str, v2: bool = False) -> list[str]:
    """Romanize adjacent Hangul syllables by sound, one string per syllable (``v2``: ㅢ rules)."""
    syllables = [decompose(c) for c in text]
    for cur, nxt in zip(syllables, syllables[1:]):
        _join(cur, nxt, v2)
    if syllables:
        syllables[-1][2] = NEUTRAL[syllables[-1][2]]
    result = []
    for index, (onset, vowel, coda) in enumerate(syllables):
        # ㄹ after an ㄹ coda is a lateral: 신라 → sil+la.
        roman_onset = (
            "l"
            if onset == "ㄹ" and index and syllables[index - 1][2] == "ㄹ"
            else ONSET_ROMAN[onset]
        )
        roman_vowel = VOWEL_ROMAN[vowel]
        if v2 and vowel == "ㅢ":
            if onset != "ㅇ":
                roman_vowel = "i"
            elif index:
                roman_vowel = "e" if index == len(syllables) - 1 else "i"
        result.append(roman_onset + roman_vowel + CODA_ROMAN[coda])
    return result


def apply_to_items(items: list[dict], v2: bool = False) -> list[dict]:
    """Replace spelling-based prons of adjacent single-syllable Hangul items."""
    items = [dict(item) for item in items]
    index = 0
    while index < len(items):
        if not is_syllable(str(items[index].get("orig", ""))):
            index += 1
            continue
        end = index
        while end < len(items) and is_syllable(str(items[end].get("orig", ""))):
            end += 1
        for item, pron in zip(
            items[index:end], romanize_run("".join(i["orig"] for i in items[index:end]), v2)
        ):
            item["pron"] = pron
            item["pronunciation"] = VERSION_V2 if v2 else VERSION
        index = end
    return items
