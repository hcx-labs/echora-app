"""Bracketed lyrics are backing vocals: the rule shared by the karaoke store and the lyric exports."""

from __future__ import annotations

BACKING_VOICE = 2  # syllable["voice"]: a second voice, shown as its own smaller line

BRACKET_OPEN, BRACKET_CLOSE = "(（", ")）"


def _bracket_counts(texts: list[str]) -> list[tuple[int, int]]:
    """Per text, how many of its letters and digits sit (inside, outside) round brackets; a
    bracket opened in one text stays open in the next, until it closes or the texts end."""
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
    """Bracketed words are backing vocals, always: every syllable written inside round brackets
    gets ``"voice": 2``, and a line that is bracketed as a whole (with or without syllables) gets
    ``"voice": "background"``. Marks other than these are left as they are."""
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
