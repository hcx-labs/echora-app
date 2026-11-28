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


def _half(anchor_ms, lead_b_offset=2000):
    """Lead, its echo, a second lead, its echo, then the clearly heard anchor line."""
    return [
        _line("o ranjhna", anchor_ms - 4500, anchor_ms - 2300, [0.2, 0.2]),
        _line("(o ranjhna)", anchor_ms - 3500, anchor_ms - 2400, [0.1, 0.1], backing=True),
        _line("bol sajna", anchor_ms - lead_b_offset, anchor_ms - 100, [0.1, 0.1]),
        _line("(bol sajna)", anchor_ms - 1200, anchor_ms - 100, [0.1, 0.1], backing=True),
        _line("tu mera", anchor_ms, anchor_ms + 4000, [0.2, 0.5]),
    ]


def _document(*halves):
    return {"alignment": {"lines": [line for half in halves for line in half]}}


def test_a_repeat_that_disagrees_with_its_twins_takes_their_timing():
    outlier = _half(40000, lead_b_offset=4100)
    # The decoder pulled the second lead line onto the first one's last word.
    outlier[0]["end_ms"] = outlier[2]["start_ms"]
    document = _document(_half(10000), _half(20000), outlier, _half(30000))
    changes = apply_repeat_template(document)
    assert [change["line"] for change in changes] == [14]
    lines = document["alignment"]["lines"]
    assert lines[12]["start_ms"] == 40000 - 2000
    assert lines[10]["end_ms"] == 40000 - 2300
    # Tokens follow the agreeing template's layout and stay in order.
    assert [t["start_ms"] for t in lines[12]["tokens"]] == [38000, 38950]
    # The agreeing halves are untouched.
    assert lines[2]["start_ms"] == 10000 - 2000


def test_nothing_changes_when_the_repeats_agree_or_are_too_few():
    agreeing = _document(_half(10000), _half(20000), _half(30000))
    before = copy.deepcopy(agreeing)
    assert apply_repeat_template(agreeing) == [] and agreeing == before
    # Two agreeing and one disagreeing is a majority; one against one is not.
    pair = _document(_half(10000), _half(20000, lead_b_offset=4100))
    assert apply_repeat_template(pair) == []


def test_an_anchor_the_model_does_not_hear_clearly_is_no_anchor():
    halves = [_half(10000), _half(20000), _half(30000, lead_b_offset=4100), _half(40000)]
    for half in halves:
        half[-1]["tokens"][1]["ctc_score"] = 0.3
    assert apply_repeat_template(_document(*halves)) == []


def test_blocks_of_another_shape_are_not_compared():
    odd = _half(30000, lead_b_offset=4100)
    odd[1]["tokens"].append(dict(odd[1]["tokens"][-1]))
    document = _document(_half(10000), _half(20000), odd)
    # The shape differs right before the anchor's lines, so no block is shared at all.
    assert apply_repeat_template(document) == []
