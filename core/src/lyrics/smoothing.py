"""Smoother karaoke timing inside words, for lyric exports.

The aligner's syllable boundaries are kept as stored; exports can smooth them on the way out.
Each step works inside one word and keeps the word's start and end, so line timing and the
gaps between words never change:

- ``close_gaps``: a syllable runs until the next one in its word starts, so the fill does not
  stop and jump inside a word.
- ``even_pacing``: a word whose syllables are very uneven (the longest more than
  ``UNEVEN_RATIO`` times the shortest) and whose longest syllable is not its last (a held note
  usually is) has its time shared out by letter count.
- ``min_syllable``: a syllable shorter than ``MIN_SYLLABLE_MS`` borrows time from its neighbours
  in the word (their gap first, then the part of a neighbour above the minimum).

A word ends at a syllable whose text ends in whitespace (perhaps followed by an opening quote
or bracket for the next word), where the voice or the aligner stream changes, and around every
syllable in a script written without spaces (CJK, Thai, ...), where syllables cannot be grouped
into words. Words with a syllable missing its timing are left alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MIN_SYLLABLE_MS = 100
UNEVEN_RATIO = 4.0

_NO_SPACES = re.compile("[฀-໿က-႟ក-៿぀-ヿ㐀-䶿一-鿿豈-﫿]")
_LETTER = re.compile(r"\w")
# Whitespace, then only punctuation glued on for the next word (an opening quote or bracket).
_WORD_END = re.compile(r"\s[^\w\s]*$")


def _timed(syllable: dict) -> bool:
    start, end = syllable.get("start_ms"), syllable.get("end_ms")
    return (
        isinstance(start, (int, float))
        and isinstance(end, (int, float))
        and not isinstance(start, bool)
        and not isinstance(end, bool)
        and 0 <= start <= end
    )


def _words(syllables: list[dict]) -> Iterable[list[dict]]:
    word: list[dict] = []
    for syllable in syllables:
        text = str(syllable.get("text") or "")
        if word and (
            _NO_SPACES.search(text)
            or syllable.get("voice") != word[-1].get("voice")
            or syllable.get("stream") != word[-1].get("stream")
        ):
            yield word
            word = []
        word.append(syllable)
        if not text or _WORD_END.search(text) or _NO_SPACES.search(text):
            yield word
            word = []
    if word:
        yield word


def _close_gaps(word: list[dict]) -> None:
    for current, following in zip(word, word[1:]):
        if current["end_ms"] < following["start_ms"]:
            current["end_ms"] = following["start_ms"]


def _even_pacing(word: list[dict]) -> None:
    durations = [s["end_ms"] - s["start_ms"] for s in word]
    longest = max(durations)
    if durations.index(longest) == len(word) - 1:
        return
    if longest <= UNEVEN_RATIO * max(1, min(durations)):
        return
    start, end = word[0]["start_ms"], word[-1]["end_ms"]
    weights = [max(1, len(_LETTER.findall(str(s.get("text") or "")))) for s in word]
    total, elapsed = sum(weights), 0
    for syllable, weight in zip(word, weights):
        syllable["start_ms"] = start + round((end - start) * elapsed / total)
        elapsed += weight
        syllable["end_ms"] = start + round((end - start) * elapsed / total)


def _min_syllable(word: list[dict]) -> None:
    for index, syllable in enumerate(word):
        deficit = MIN_SYLLABLE_MS - (syllable["end_ms"] - syllable["start_ms"])
        if deficit > 0 and index + 1 < len(word):
            following = word[index + 1]
            room = (following["start_ms"] - syllable["end_ms"]) + max(
                0, following["end_ms"] - following["start_ms"] - MIN_SYLLABLE_MS
            )
            take = min(deficit, max(0, room))
            syllable["end_ms"] += take
            following["start_ms"] = max(following["start_ms"], syllable["end_ms"])
            deficit -= take
        if deficit > 0 and index > 0:
            previous = word[index - 1]
            room = (syllable["start_ms"] - previous["end_ms"]) + max(
                0, previous["end_ms"] - previous["start_ms"] - MIN_SYLLABLE_MS
            )
            take = min(deficit, max(0, room))
            syllable["start_ms"] -= take
            previous["end_ms"] = min(previous["end_ms"], syllable["start_ms"])


def smooth_karaoke(
    lines: list[dict],
    *,
    close_gaps: bool = False,
    even_pacing: bool = False,
    min_syllable: bool = False,
) -> list[dict]:
    """The lines with smoothed syllable timing; the input is not changed."""
    if not (close_gaps or even_pacing or min_syllable):
        return lines
    result = []
    for line in lines:
        syllables = line.get("syllables") if isinstance(line, dict) else None
        if not isinstance(syllables, list) or not syllables:
            result.append(line)
            continue
        copied = [dict(s) if isinstance(s, dict) else s for s in syllables]
        if all(isinstance(s, dict) for s in copied):
            for word in _words(copied):
                if len(word) < 2 or not all(_timed(s) for s in word):
                    continue
                if close_gaps:
                    _close_gaps(word)
                if even_pacing:
                    _even_pacing(word)
                if min_syllable:
                    _min_syllable(word)
        result.append({**line, "syllables": copied})
    return result
