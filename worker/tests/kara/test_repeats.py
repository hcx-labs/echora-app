import copy

from worker.kara.aligner.repeats import apply_repeat_template


def _line(text, start, end, scores, backing=False):
    """A line whose tokens split [start, end] evenly, one per score."""
    step = (end - start) / len(scores)
    tokens = [
        {
            "text": f"{text}{i}",
            "start_ms": int(start + i * step),
            "end_ms": int(start + (i + 1) * step),
            "ctc_score": score,
            **({"stream": 2} if backing else {}),
        }
        for i, score in enumerate(scores)
    ]
    return {"text": text, "start_ms": start, "end_ms": end, "tokens": tokens}


def _chorus(at, second=(2000, 3800), words=("o ranjhna", "bol sajna")):
    """A heard line, two poorly heard lines and an echo, starting at ``at`` ms."""
    return [
        _line("tu mera", at, at + 1800, [0.2, 0.6]),
        _line(words[0], at + 2000, at + 3800, [0.2, 0.2]),
        _line(f"({words[0]})", at + 2800, at + 3700, [0.1, 0.1], backing=True),
        _line(words[1], at + second[0] + 2000, at + second[1] + 2000, [0.1, 0.1]),
    ]


def _verse(at):
    return [_line(f"verse {at}", at, at + 3000, [0.5, 0.5])]


def _document(*parts):
    return {"alignment": {"lines": [line for part in parts for line in part]}}


def test_a_line_that_disagrees_with_its_twins_takes_their_timing():
    # The third chorus put its last line onto the line before it.
    document = _document(
        _chorus(0), _verse(8000), _chorus(12000), _verse(20000), _chorus(24000, second=(600, 1800))
    )
    changes = [c for c in apply_repeat_template(document) if "after_ms" in c]
    assert [(c["line"], c["section"]) for c in changes] == [(13, "text")]
    line = document["alignment"]["lines"][13]
    assert (line["start_ms"], line["end_ms"]) == (24000 + 4000, 24000 + 5800)
    assert [t["start_ms"] for t in line["tokens"]] == [28000, 28900]


def test_repeats_that_rhyme_other_words_to_one_rhythm_are_compared_by_shape():
    document = _document(
        _chorus(0),
        _chorus(10000, words=("bol ranjhna", "o sajna")),
        _chorus(20000, second=(600, 1800)),
        _chorus(30000, words=("bol ranjhna", "o sajna")),
    )
    changes = [c for c in apply_repeat_template(document) if "after_ms" in c]
    assert [(c["line"], c["section"]) for c in changes] == [(11, "shape")]


def test_nothing_changes_when_repeats_agree_or_too_few_agree():
    agreeing = _document(_chorus(0), _chorus(10000), _chorus(20000))
    before = copy.deepcopy(agreeing)
    assert apply_repeat_template(agreeing) == [] and agreeing == before
    split = _document(_chorus(0), _chorus(10000, second=(600, 1800)), _chorus(20000))
    split["alignment"]["lines"][3]["start_ms"] += 900  # only one twin left that agrees
    split["alignment"]["lines"][3]["end_ms"] += 900
    assert [c for c in apply_repeat_template(split) if "after_ms" in c] == []


def test_clearly_heard_lines_are_never_moved_and_unheard_repeats_are_not_placed():
    heard = _document(_chorus(0), _chorus(10000), _chorus(20000, second=(600, 1800)))
    heard["alignment"]["lines"][11]["tokens"][1]["ctc_score"] = 0.8
    assert [c for c in apply_repeat_template(heard) if "after_ms" in c] == []
    unheard = _document(_chorus(0), _chorus(10000), _chorus(20000, second=(600, 1800)))
    for line in unheard["alignment"]["lines"]:
        for token in line["tokens"]:
            token["ctc_score"] = 0.1
    assert apply_repeat_template(unheard) == []


def test_a_line_never_moves_past_its_neighbours():
    # A verse line sung where the third chorus's last line belongs.
    document = _document(
        _chorus(0), _chorus(10000), _chorus(20000, second=(600, 1800)), _verse(23000)
    )
    changes = apply_repeat_template(document)
    assert {"line": 11, "skipped": "would pass a neighbour"} in changes


def test_backing_lines_marked_on_the_line_are_backing():
    # The decoder marks whole backing lines with ``voice``, not each token's stream.
    from worker.kara.aligner.repeats import _is_backing_line

    echo = _line("(o ranjhna)", 0, 1000, [0.1, 0.1])
    assert not _is_backing_line(echo)
    echo["voice"] = "background"
    assert _is_backing_line(echo)
