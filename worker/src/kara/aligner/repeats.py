"""Repeat template: repeats of a block that disagree with the majority take the majority's timing.

Choruses and hooks repeat, usually to the same rhythm. Where the model hears little (legato
singing, held vowels), the decoder can place one repeat very differently from its twins: Wafa's
second extended chorus put "بول ساجنا" 2 s early, on the one "jana" the model heard, while three
other halves of the same chorus agreed on where the line falls. A confidence rule would not have
caught it: the misplaced word scored higher than its correctly placed twins.

A block is anchored by a line the model hears clearly every time it is sung (a word scoring at
least ``ANCHOR_SCORE``), which recurs at least ``MIN_REPEATS`` times. The lines just before each
occurrence are compared by shape (lead or backing line, token counts), as long as all occurrences
share it. Each line's start is measured back from its anchor; an occurrence whose lines sit more
than ``TOLERANCE_MS`` from the majority's is re-timed from the median of the occurrences that
agree, token by token after the agreeing occurrence closest to that median.

Off by default (``KaraConfig.repeat_template``); runs on the finished alignment document, so a
replayed reading applies it without the model.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

ANCHOR_SCORE = 0.45
MIN_REPEATS = 3
TOLERANCE_MS = 600
SPAN_TOLERANCE = 0.25
MAX_BLOCK_LINES = 8
SECOND_STREAM = 2

Line = dict[str, Any]


def _normalized(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _is_backing_line(line: Line) -> bool:
    tokens = line.get("tokens") or []
    return bool(tokens) and all(token.get("stream") == SECOND_STREAM for token in tokens)


def _lead_score(line: Line) -> float:
    scores = [
        float(token.get("ctc_score") or 0)
        for token in line.get("tokens") or []
        if token.get("stream") != SECOND_STREAM
    ]
    return max(scores, default=0.0)


def _shape(line: Line) -> tuple[bool, int, int]:
    tokens = line.get("tokens") or []
    backing = sum(1 for token in tokens if token.get("stream") == SECOND_STREAM)
    return (_is_backing_line(line), len(tokens), backing)


def _timed(line: Line) -> bool:
    tokens = line.get("tokens") or []
    return (
        bool(tokens)
        and isinstance(line.get("start_ms"), int | float)
        and all(isinstance(t.get("start_ms"), int | float) for t in tokens)
    )


def _anchors(lines: list[Line]) -> dict[str, list[int]]:
    """Clearly heard lead lines whose text recurs, by text."""
    heard: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        if _timed(line) and not _is_backing_line(line) and _lead_score(line) >= ANCHOR_SCORE:
            heard.setdefault(_normalized(str(line.get("text") or "")), []).append(index)
    return {
        text: indices for text, indices in heard.items() if text and len(indices) >= MIN_REPEATS
    }


def _block_length(lines: list[Line], anchors: list[int], anchor_set: set[int]) -> int:
    """How many lines before every occurrence share one shape (never crossing an anchor)."""
    length = 0
    while length < MAX_BLOCK_LINES:
        position = length + 1
        shapes = set()
        for anchor in anchors:
            index = anchor - position
            if index < 0 or index in anchor_set or not _timed(lines[index]):
                return length
            shapes.add(_shape(lines[index]))
        if len(shapes) != 1:
            return length
        length += 1
    return length


def _retime(template: Line, target: Line, start_ms: float, end_ms: float) -> None:
    """Lay ``target``'s tokens over [start, end] the way ``template``'s fill its own line."""
    t_start, t_end = float(template["start_ms"]), float(template["end_ms"])
    span = max(t_end - t_start, 1.0)
    scale = (end_ms - start_ms) / span
    for source, token in zip(template["tokens"], target["tokens"], strict=True):
        token["start_ms"] = int(round(start_ms + (float(source["start_ms"]) - t_start) * scale))
        token["end_ms"] = int(round(start_ms + (float(source["end_ms"]) - t_start) * scale))
        token.pop("acoustic_start_ms", None)
        token.pop("acoustic_end_ms", None)
    target["start_ms"] = min(int(t["start_ms"]) for t in target["tokens"])
    target["end_ms"] = max(int(t["end_ms"]) for t in target["tokens"])


def apply_repeat_template(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-time repeats that disagree with their twins, in place; returns what changed."""
    alignment = document.get("alignment") or {}
    lines: list[Line] = alignment.get("lines") or []
    changes: list[dict[str, Any]] = []
    anchors = _anchors(lines)
    anchor_set = {index for indices in anchors.values() for index in indices}
    for text, occurrences in anchors.items():
        length = _block_length(lines, occurrences, anchor_set)
        if length == 0:
            continue
        # Offsets back from each anchor: start and end of every block line.
        offsets = {
            anchor: [
                (
                    float(lines[anchor]["start_ms"]) - float(lines[anchor - p]["start_ms"]),
                    float(lines[anchor]["start_ms"]) - float(lines[anchor - p]["end_ms"]),
                )
                for p in range(length, 0, -1)
            ]
            for anchor in occurrences
        }
        medians = [statistics.median(offsets[a][i][0] for a in occurrences) for i in range(length)]
        agree = [
            a
            for a in occurrences
            if all(abs(offsets[a][i][0] - medians[i]) <= TOLERANCE_MS for i in range(length))
        ]
        disagree = [a for a in occurrences if a not in agree]
        if not disagree or len(agree) < 2 or len(agree) <= len(disagree):
            continue
        starts = [statistics.median(offsets[a][i][0] for a in agree) for i in range(length)]
        ends = [statistics.median(offsets[a][i][1] for a in agree) for i in range(length)]
        span = starts[0]
        if any(abs(offsets[a][0][0] - span) > SPAN_TOLERANCE * span for a in disagree):
            # The block itself is of another length there (another arrangement): leave it.
            continue
        closest = min(
            agree, key=lambda a: sum(abs(offsets[a][i][0] - starts[i]) for i in range(length))
        )
        for anchor in disagree:
            anchor_ms = float(lines[anchor]["start_ms"])
            first = anchor - length
            previous_lead = next(
                (
                    lines[i]
                    for i in range(first - 1, -1, -1)
                    if _timed(lines[i]) and not _is_backing_line(lines[i])
                ),
                None,
            )
            new_start = anchor_ms - starts[0]
            if previous_lead is not None and new_start < float(previous_lead["start_ms"]):
                changes.append({"anchor": text, "line": anchor, "skipped": "before previous line"})
                continue
            before = [
                (int(lines[first + i]["start_ms"]), int(lines[first + i]["end_ms"]))
                for i in range(length)
            ]
            for i in range(length):
                start_ms = anchor_ms - starts[i]
                end_ms = min(anchor_ms - ends[i], anchor_ms)
                if end_ms <= start_ms:
                    end_ms = start_ms + 1
                _retime(lines[closest - length + i], lines[first + i], start_ms, end_ms)
            changes.append(
                {
                    "anchor": text,
                    "line": anchor,
                    "lines": list(range(first, anchor)),
                    "template": closest,
                    "agreeing": len(agree),
                    "before_ms": before,
                    "after_ms": [
                        (int(lines[first + i]["start_ms"]), int(lines[first + i]["end_ms"]))
                        for i in range(length)
                    ],
                }
            )
    return changes
