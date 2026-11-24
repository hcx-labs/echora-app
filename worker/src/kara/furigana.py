"""Furigana for Japanese karaoke lines, built from the aligner's own units.

The aligner times a kanji group (彼女) as one display token followed by silent continuation
units, each with its own romanized mora and timing (ka, no, jo). Converting those morae back to
hiragana gives a reading that always matches what was timed, mora by mora. Kana inside a group
(the び of 飛び交) anchor the reading, so only the kanji get ruby.
"""

from __future__ import annotations

import re

_KANJI = re.compile(r"[㐀-䶿一-鿿豈-﫿々〆]")
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")

_VOWELS = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}
_ROWS = {
    "k": "かきくけこ",
    "g": "がぎぐげご",
    "s": "さしすせそ",
    "z": "ざじずぜぞ",
    "t": "たちつてと",
    "d": "だぢづでど",
    "n": "なにぬねの",
    "h": "はひふへほ",
    "b": "ばびぶべぼ",
    "p": "ぱぴぷぺぽ",
    "m": "まみむめも",
    "r": "らりるれろ",
}
_SYLLABLES: dict[str, str] = dict(_VOWELS)
for consonant, row in _ROWS.items():
    _SYLLABLES.update({consonant + vowel: kana for vowel, kana in zip("aiueo", row)})
_SYLLABLES.update(
    {
        "shi": "し",
        "chi": "ち",
        "tsu": "つ",
        "fu": "ふ",
        "ji": "じ",
        "zu": "ず",
        "ya": "や",
        "yu": "ゆ",
        "yo": "よ",
        "wa": "わ",
        "wo": "を",
        "n": "ん",
        "fa": "ふぁ",
        "fi": "ふぃ",
        "fe": "ふぇ",
        "fo": "ふぉ",
        "ti": "てぃ",
        "di": "でぃ",
        "she": "しぇ",
        "che": "ちぇ",
        "je": "じぇ",
        "va": "ゔぁ",
        "vi": "ゔぃ",
        "vu": "ゔ",
    }
)
for consonant, base in (
    ("k", "き"),
    ("g", "ぎ"),
    ("n", "に"),
    ("h", "ひ"),
    ("b", "び"),
    ("p", "ぴ"),
    ("m", "み"),
    ("r", "り"),
):
    _SYLLABLES.update({consonant + "y" + v: base + small for v, small in zip("auo", "ゃゅょ")})
for prefix, base in (("sh", "し"), ("ch", "ち"), ("j", "じ")):
    _SYLLABLES.update({prefix + v: base + small for v, small in zip("auo", "ゃゅょ")})


_SOKUON = set("kgsztdhbpmrcfj")


def romaji_to_hiragana(romaji: str) -> str | None:
    """Hiragana for one Hepburn unit (``ka``, ``nat`` → なっ, ``itta`` → いった); None if unknown."""
    text, out, i = romaji.lower(), [], 0
    while i < len(text):
        # A doubled consonant (itta) or a unit-final consonant (nat) is a sokuon.
        if text[i] in _SOKUON and (
            i + 1 == len(text)
            or text[i + 1] == text[i]
            or (text[i] == "t" and text[i + 1 : i + 3] == "ch")
        ):
            out.append("っ")
            i += 1
            continue
        for size in (3, 2, 1):
            kana = _SYLLABLES.get(text[i : i + size])
            # A bare n is ん only before a consonant or at the end (na is な, not ん + あ).
            if kana and not (
                size == 1 and text[i] == "n" and i + 1 < len(text) and text[i + 1] in "aiueoy"
            ):
                out.append(kana)
                i += size
                break
        else:
            return None
    return "".join(out)


def _hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def ruby_segments(surface: str, morae: list[dict[str, object]]) -> list[dict[str, object]]:
    """Ruby over the kanji runs of one display token.

    ``morae`` are the token's aligner units in order: ``{"unit": romaji, "start_ms", "end_ms"}``.
    Returns ``[{"start", "end", "text", "morae": [{"text", "start_ms", "end_ms"}]}]`` with
    ``start``/``end`` as character offsets into ``surface``; empty when there is nothing to annotate
    or the reading cannot be placed.
    """
    if not _KANJI.search(surface) or not morae:
        return []
    kana_of_mora = [romaji_to_hiragana(str(mora.get("unit") or "")) for mora in morae]
    if any(not kana for kana in kana_of_mora):
        return []
    reading = "".join(kana_of_mora)
    owner = [index for index, kana in enumerate(kana_of_mora) for _ in kana]
    plain = _hiragana(surface)
    segments: list[dict[str, object]] = []
    position = cursor = 0
    while position < len(surface):
        if _KANA.match(surface[position]):
            if cursor >= len(reading) or reading[cursor] != plain[position]:
                break  # the kana anchor is not where the reading says: give up on placement
            position, cursor = position + 1, cursor + 1
            continue
        end = position
        while end < len(surface) and not _KANA.match(surface[end]):
            end += 1
        if end == len(surface):
            stop = len(reading)
        else:
            stop = reading.find(plain[end], cursor + 1)
            if stop < 0:
                break
        if stop <= cursor:
            break
        if _KANJI.search(surface[position:end]):
            used = sorted(set(owner[cursor:stop]))
            pieces = []
            for index in used:
                first = max(cursor, owner.index(index))
                last = min(stop, len(owner) - owner[::-1].index(index))
                pieces.append(
                    {
                        "text": reading[first:last],
                        "start_ms": int(morae[index]["start_ms"]),
                        "end_ms": int(morae[index]["end_ms"]),
                    }
                )
            segments.append(
                {"start": position, "end": end, "text": reading[cursor:stop], "morae": pieces}
            )
        position, cursor = end, stop
    else:
        if cursor == len(reading):
            return segments
    # Kana anchors did not line up: one reading over the whole token is still correct.
    return [
        {
            "start": 0,
            "end": len(surface),
            "text": reading,
            "morae": [
                {"text": kana, "start_ms": int(mora["start_ms"]), "end_ms": int(mora["end_ms"])}
                for kana, mora in zip(kana_of_mora, morae)
            ],
        }
    ]
