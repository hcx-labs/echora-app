"""Bracketed lyrics are backing vocals: the rule shared by the karaoke store and the lyric exports."""

from __future__ import annotations

import re

BACKING_VOICE = 2  # syllable["voice"]: a second voice, shown as its own smaller line

BRACKET_OPEN, BRACKET_CLOSE = "(（", ")）"


def _bracket_counts(texts: list[str]) -> list[tuple[int, int]]:
    """Count, per text, the letters and digits (inside, outside) round brackets.

    A bracket opened in one text stays open in the next, until it closes or the texts end.
    """
    counts, depth = [], 0
    for text in texts:
        inside = outside = 0
        for character in text:
            if character in BRACKET_OPEN:
                depth += 1
            elif character in BRACKET_CLOSE:
                depth = max(0, depth - 1)
            elif character.isalnum():
                inside += depth > 0
                outside += depth == 0
        counts.append((inside, outside))
    return counts


def tag_bracketed_backing(lines: list[dict[str, object]]) -> list[dict[str, object]]:
    """Tag bracketed words as backing vocals, always.

    Every syllable written inside round brackets gets ``"voice": 2``, and a line that is
    bracketed as a whole (with or without syllables) gets ``"voice": "background"``. Marks other
    than these are left as they are.
    """
    for line in lines:
        syllables = [s for s in line.get("syllables") or [] if isinstance(s, dict)]
        if syllables:
            counts = _bracket_counts([str(s.get("text") or "") for s in syllables])
            for syllable, (inside, outside) in zip(syllables, counts):
                if inside and inside >= outside:
                    syllable["voice"] = BACKING_VOICE
            sung = [s for s, (inside, outside) in zip(syllables, counts) if inside or outside]
            whole = bool(sung) and all(s.get("voice") == BACKING_VOICE for s in sung)
        else:
            ((inside, outside),) = _bracket_counts([str(line.get("text") or "")])
            whole = bool(inside) and not outside
        if whole:
            line["voice"] = "background"
        elif line.get("voice") == "background":
            line.pop("voice")
    return lines


_BRACKETS = re.compile(r"[()（）]")
_PUNCTUATION = ",.!?;:、。！？，"


def _end_part(part: list[dict[str, object]]) -> None:
    """Close one backing part before the next: "hey" becomes "hey, "."""
    text = str(part[-1].get("text") or "").rstrip()
    if text and text[-1] not in _PUNCTUATION:
        text += ","
    part[-1] = {**part[-1], "text": text + " "}


def _add(part: list[dict[str, object]], syllable: dict[str, object], text: str) -> None:
    text = re.sub(r"\s+", " ", text)
    if not part or str(part[-1]["text"]).endswith(" "):
        text = text.lstrip()
    if part and text[:1] in tuple(_PUNCTUATION) and str(part[-1]["text"]).endswith(" "):
        # "far away (…), is" loses the bracket: no space before the comma.
        part[-1] = {**part[-1], "text": str(part[-1]["text"]).rstrip()}
    if not text.strip():
        # A bracket alone, or a gap: keep the word break, drop the syllable.
        if text and part and not str(part[-1]["text"]).endswith(" "):
            part[-1] = {**part[-1], "text": str(part[-1]["text"]) + " "}
        return
    part.append({**syllable, "text": text})


def separate_voices(
    syllables: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """A line's lead syllables and its backing syllables (``voice: 2``), without brackets.

    "Hello (hey) I am (hey)" becomes lead "Hello I am" and backing "hey, hey": each bracketed
    part (or each run of backing syllables between lead words) is one backing part, and parts
    are joined with a comma. Timing stays on each syllable. Run ``tag_bracketed_backing`` first
    so bracketed words carry ``voice: 2``.
    """
    lead: list[dict[str, object]] = []
    backing: list[dict[str, object]] = []
    new_part = False
    for syllable in syllables:
        raw = str(syllable.get("text") or "")
        if syllable.get("voice") == BACKING_VOICE:
            opens = any(c in BRACKET_OPEN for c in raw)
            if backing and (new_part or opens):
                _end_part(backing)
            new_part = any(c in BRACKET_CLOSE for c in raw)
            _add(backing, syllable, _BRACKETS.sub("", raw))
        else:
            # Lead words between two backing runs split them into separate parts.
            new_part = new_part or bool(backing)
            _add(lead, syllable, _BRACKETS.sub("", raw))
    for part in (lead, backing):
        if part:
            part[-1] = {**part[-1], "text": str(part[-1]["text"]).rstrip()}
    return lead, backing


def separate_text(text: str) -> tuple[str, str]:
    """The same split for a line without word timing: ("Hello I am", "hey, hey")."""
    pieces = re.split(r"([(（][^()（）]*[)）])", text)
    if len(pieces) == 1:
        return text, ""
    syllables = [
        {"text": piece, "voice": BACKING_VOICE if index % 2 else None}
        for index, piece in enumerate(pieces)
        if piece
    ]
    lead, backing = separate_voices(syllables)
    return "".join(str(s["text"]) for s in lead), "".join(str(s["text"]) for s in backing)
