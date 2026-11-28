"""Repeat template: lines of repeated sections that disagree with their twins take their timing.

Choruses and hooks repeat, usually to the same rhythm. Where the model hears little (legato
singing, held vowels), the decoder can place one repeat very differently from the others: Wafa's
second extended chorus put "بول ساجنا" 2 s early, on the one "jana" the model heard, while three
other halves of the same chorus agreed on where the line falls. A confidence rule would not have
caught it: the misplaced word scored higher than its correctly placed twins.

Sections are runs of 2 to ``MAX_SECTION_LINES`` lines that occur at least ``MIN_REPEATS`` times:
first by text (a chorus sung the same way), then, among the lines left, by shape (lead or backing
line and token counts: Wafa's halves rhyme different words to one rhythm).

The section's template is each line's median position relative to its occurrence. Each occurrence
is placed by its clearly heard lines (a lead word scoring at least ``ANCHOR_SCORE``). A line
whose start or end sits more than ``tolerance_ms`` from where the template puts it is re-timed
there, when the occurrences that agree on that line outnumber those that do not (at least two).
Clearly heard lines are never moved, and a line never moves past its neighbours.

Off by default (``KaraConfig.repeat_template``); runs on the finished alignment document, so a
replayed reading applies it without the model.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

ANCHOR_SCORE = 0.45
MIN_REPEATS = 3
MAX_SECTION_LINES = 8
TOLERANCE_MS = 600
SECOND_STREAM = 2

Line = dict[str, Any]


def _normalized(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _is_backing_line(line: Line) -> bool:
    """A backing-vocal line: marked on the line, or every token on the second stream."""
    if line.get("voice") == "background":
        return True
    tokens = line.get("tokens") or []
    return bool(tokens) and all(token.get("stream") == SECOND_STREAM for token in tokens)


def _lead_score(line: Line) -> float:
    scores = [
        float(token.get("ctc_score") or 0)
        for token in line.get("tokens") or []
        if token.get("stream") != SECOND_STREAM
    ]
    return max(scores, default=0.0)


def _heard(line: Line, anchor_score: float) -> bool:
    return not _is_backing_line(line) and _lead_score(line) >= anchor_score


def _shape(line: Line) -> tuple[bool, int, int]:
    tokens = line.get("tokens") or []
    backing = sum(1 for token in tokens if token.get("stream") == SECOND_STREAM)
    return (_is_backing_line(line), len(tokens), backing)


def _timed(line: Line) -> bool:
    tokens = line.get("tokens") or []
    return (
        bool(tokens)
        and isinstance(line.get("start_ms"), int | float)
        and isinstance(line.get("end_ms"), int | float)
        and all(isinstance(t.get("start_ms"), int | float) for t in tokens)
    )


def _sections(lines: list[Line], key, taken: set[int]) -> list[tuple[int, list[int]]]:
    """Runs of lines repeated at least ``MIN_REPEATS`` times by ``key``, longest first.

    Each line belongs to at most one section. Returns (length, occurrence starts).
    """
    keys = [key(line) if _timed(line) else None for line in lines]
    found = []
    for length in range(MAX_SECTION_LINES, 1, -1):
        runs: dict[tuple, list[int]] = {}
        for start in range(len(lines) - length + 1):
            span = range(start, start + length)
            if any(i in taken or keys[i] is None for i in span):
                continue
            runs.setdefault(tuple(keys[i] for i in span), []).append(start)
        for starts in runs.values():
            chosen: list[int] = []
            for start in starts:  # non-overlapping, earliest first
                if not chosen or start >= chosen[-1] + length:
                    chosen.append(start)
            chosen = [s for s in chosen if not any(i in taken for i in range(s, s + length))]
            if len(chosen) >= MIN_REPEATS:
                found.append((length, chosen))
                taken.update(i for s in chosen for i in range(s, s + length))
    return found


def _retime(template: Line, target: Line, start_ms: float, end_ms: float) -> None:
    """Lay ``target``'s tokens over [start, end] the way ``template``'s fill its own line."""
    source = template if len(template["tokens"]) == len(target["tokens"]) else target
    t_start, t_end = float(source["start_ms"]), float(source["end_ms"])
    scale = (end_ms - start_ms) / max(t_end - t_start, 1.0)
    layout = [(float(t["start_ms"]), float(t["end_ms"])) for t in source["tokens"]]
    for (s, e), token in zip(layout, target["tokens"], strict=True):
        token["start_ms"] = int(round(start_ms + (s - t_start) * scale))
        token["end_ms"] = int(round(start_ms + (e - t_start) * scale))
        token.pop("acoustic_start_ms", None)
        token.pop("acoustic_end_ms", None)
    target["start_ms"] = int(round(start_ms))
    target["end_ms"] = int(round(end_ms))


def _neighbours_allow(lines: list[Line], index: int, start_ms: float) -> bool:
    """A lead line stays between the lead lines around it (backing lines may overlap)."""
    if _is_backing_line(lines[index]):
        return True
    for step in (-1, 1):
        i = index + step
        while 0 <= i < len(lines) and (not _timed(lines[i]) or _is_backing_line(lines[i])):
            i += step
        if 0 <= i < len(lines):
            other = float(lines[i]["start_ms"])
            if (step < 0 and start_ms < other) or (step > 0 and start_ms > other):
                return False
    return True


def _end_before_next(lines: list[Line], index: int, end_ms: float) -> float:
    """A lead line's end, no later than the next lead line's start."""
    if _is_backing_line(lines[index]):
        return end_ms
    for line in lines[index + 1 :]:
        if _timed(line) and not _is_backing_line(line):
            return min(end_ms, float(line["start_ms"]))
    return end_ms


def _repair(
    lines: list[Line],
    length: int,
    starts: list[int],
    tolerance_ms: float,
    anchor_score: float,
    by: str,
) -> list[dict[str, Any]]:
    begin = {s: [float(lines[s + p]["start_ms"]) for p in range(length)] for s in starts}
    end = {s: [float(lines[s + p]["end_ms"]) for p in range(length)] for s in starts}
    # Template: positions relative to each occurrence's median line start.
    centre = {s: statistics.median(begin[s]) for s in starts}
    t_begin = [statistics.median(begin[s][p] - centre[s] for s in starts) for p in range(length)]
    t_end = [statistics.median(end[s][p] - centre[s] for s in starts) for p in range(length)]
    # Each occurrence placed by its clearly heard lines.
    place = {}
    for s in starts:
        heard = [p for p in range(length) if _heard(lines[s + p], anchor_score)]
        if not heard:
            continue
        at = statistics.median(begin[s][p] - t_begin[p] for p in heard)
        # Heard lines that disagree among themselves: this repeat is arranged otherwise.
        if all(abs(begin[s][p] - t_begin[p] - at) <= tolerance_ms for p in heard):
            place[s] = at
    if len(place) < 2:
        return []
    changes = []

    def record(index, line, before, kind, agreeing, template):
        changes.append(
            {
                "line": index,
                "text": line.get("text"),
                "section": by,
                "moved": kind,
                "repeats": len(starts),
                "agreeing": agreeing,
                "template": template,
                "before_ms": before,
                "after_ms": (int(line["start_ms"]), int(line["end_ms"])),
            }
        )

    fixed: dict[int, set[int]] = {p: set() for p in range(length)}
    for p in range(length):
        # Starts first: a line sung somewhere else takes the agreeing repeats' start and end.
        dev = {s: abs(begin[s][p] - place[s] - t_begin[p]) for s in place}
        agree = [s for s in place if dev[s] <= tolerance_ms]
        disagree = [s for s in place if dev[s] > tolerance_ms]
        if disagree and len(agree) >= 2 and len(agree) > len(disagree):
            new_begin = statistics.median(begin[s][p] - place[s] for s in agree)
            new_end = statistics.median(end[s][p] - place[s] for s in agree)
            closest = min(agree, key=lambda s: dev[s])
            for s in disagree:
                index, line = s + p, lines[s + p]
                if _heard(line, anchor_score):
                    continue
                start_ms = place[s] + new_begin
                end_ms = _end_before_next(lines, index, place[s] + max(new_end, new_begin + 1))
                if not _neighbours_allow(lines, index, start_ms) or end_ms <= start_ms:
                    changes.append({"line": index, "skipped": "would pass a neighbour"})
                    continue
                before = (int(line["start_ms"]), int(line["end_ms"]))
                _retime(lines[closest + p], line, start_ms, end_ms)
                fixed[p].add(s)
                record(index, line, before, "start", len(agree), closest + p)
    for p in range(length):
        # Then ends alone (held notes vary more, so they are judged on their own): a line that
        # starts where its twins do but ends far from them takes their end.
        rest = [s for s in place if s not in fixed[p]]
        dev = {s: abs(end[s][p] - place[s] - t_end[p]) for s in rest}
        agree = [s for s in rest if dev[s] <= tolerance_ms] + sorted(fixed[p])
        disagree = [s for s in rest if dev[s] > tolerance_ms]
        if disagree and len(agree) >= 2 and len(agree) > len(disagree):
            new_end = statistics.median(end[s][p] - place[s] for s in agree)
            closest = min(
                (s for s in agree if s not in fixed[p]), key=lambda s: dev[s], default=None
            )
            for s in disagree:
                index, line = s + p, lines[s + p]
                if _heard(line, anchor_score) or closest is None:
                    continue
                start_ms = float(line["start_ms"])
                end_ms = _end_before_next(lines, index, place[s] + new_end)
                if end_ms <= start_ms:
                    continue
                before = (int(line["start_ms"]), int(line["end_ms"]))
                _retime(lines[closest + p], line, start_ms, end_ms)
                record(index, line, before, "end", len(agree), closest + p)
    return changes


def apply_repeat_template(
    document: dict[str, Any],
    *,
    tolerance_ms: float = TOLERANCE_MS,
    anchor_score: float = ANCHOR_SCORE,
) -> list[dict[str, Any]]:
    """Re-time lines of repeated sections that disagree with their twins, in place.

    Returns what changed, and what was left alone and why.
    """
    alignment = document.get("alignment") or {}
    lines: list[Line] = alignment.get("lines") or []
    taken: set[int] = set()
    changes: list[dict[str, Any]] = []
    for by, key in (
        ("text", lambda line: (_is_backing_line(line), _normalized(str(line.get("text") or "")))),
        ("shape", _shape),
    ):
        for length, starts in _sections(lines, key, taken):
            changes += _repair(lines, length, starts, tolerance_ms, anchor_score, by)
    return changes
