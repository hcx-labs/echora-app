import sys
from pathlib import Path

import torch

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "fa_kara"
sys.path.insert(0, str(VENDOR))

from ctc_segmentation import SourcePrior, align_with_source_priors  # noqa: E402
from align_yohane import (  # noqa: E402
    _assign_ctc_blank_holds,
    _inference_chunk_starts,
)


def test_ctc_blank_frames_extend_the_preceding_acoustic_token():
    from torchaudio.functional import TokenSpan

    spans = [
        [TokenSpan(1, 2, 4, 0.8)],
        [TokenSpan(2, 9, 11, 0.7)],
        [TokenSpan(3, 15, 17, 0.9)],
    ]

    result = _assign_ctc_blank_holds(spans, [(0, 2), (2, 3)])

    assert result[0][-1].end == 9
    assert result[1][-1].end == 11
    assert result[2][-1].end == 17
    assert spans[0][-1].end == 4


def _emission(labels, vocabulary=4):
    logits = torch.full((len(labels), vocabulary), -8.0)
    for frame, label in enumerate(labels):
        logits[frame, label] = 8.0
    return torch.log_softmax(logits, dim=-1)


def test_global_ctc_alignment_returns_monotonic_grouped_spans():
    emission = _emission([0, 1, 1, 0, 2, 2, 0])

    spans = align_with_source_priors(
        emission,
        [[1], [2]],
        blank=0,
        frame_shift_seconds=0.02,
        source_priors=[SourcePrior(0, 0.02), SourcePrior(1, 0.08)],
    )

    assert [(span.start, span.end) for group in spans for span in group] == [(1, 3), (4, 6)]


def test_repeated_ctc_labels_require_an_intervening_blank():
    emission = _emission([0, 1, 0, 1, 0])

    spans = align_with_source_priors(
        emission,
        [[1], [1]],
        blank=0,
        frame_shift_seconds=0.02,
        source_priors=[],
    )

    assert spans[0][0].end <= spans[1][0].start


def test_source_prior_disambiguates_repeated_acoustic_occurrences():
    # Two equally plausible occurrences of token 1. The onset prior should pick
    # the later one instead of the first path available to Viterbi.
    emission = _emission([0, 1, 0, 0, 0, 1, 0])

    spans = align_with_source_priors(
        emission,
        [[1]],
        blank=0,
        frame_shift_seconds=0.1,
        source_priors=[SourcePrior(0, 0.5)],
        prior_weight=4.0,
    )

    assert spans[0][0].start == 5


def test_inference_chunks_do_not_create_tiny_tail_passes():
    sample_rate = 16_000
    chunk = 45 * sample_rate
    overlap = 2 * sample_rate
    step = chunk - overlap

    assert _inference_chunk_starts(step * 5 + 1, chunk, overlap) == [
        0,
        step,
        step * 2,
        step * 3,
        step * 4,
    ]
    assert _inference_chunk_starts(step * 5 + overlap + 1, chunk, overlap)[-1] == step * 5


def test_impossible_alignment_fails_instead_of_returning_partial_output():
    emission = _emission([0, 1])

    try:
        align_with_source_priors(
            emission,
            [[1], [2], [3]],
            blank=0,
            frame_shift_seconds=0.02,
            source_priors=[],
        )
    except RuntimeError as error:
        assert "fewer frames" in str(error)
    else:
        raise AssertionError("expected alignment failure")


def test_minimum_frames_counts_repeats_across_group_boundaries():
    from ctc_segmentation import minimum_ctc_frames

    assert minimum_ctc_frames([]) == 0
    assert minimum_ctc_frames([[1, 2], [3]]) == 3
    assert minimum_ctc_frames([[1, 1], [1, 2]]) == 6


def test_repeated_targets_reject_windows_without_room_for_blanks():
    emission = torch.log_softmax(torch.zeros(2, 3), dim=-1)
    try:
        align_with_source_priors(
            emission, [[1, 1]], blank=0, frame_shift_seconds=0.02, source_priors=[]
        )
    except RuntimeError as error:
        assert "fewer frames" in str(error)
    else:
        raise AssertionError("Repeated labels require an intervening blank frame")


def test_minimum_durations_match_plain_ctc_at_one_frame_and_stop_cramming():
    from ctc_segmentation import align_with_minimum_durations, token_minimum_frames

    blank, a, b, c = 0, 1, 2, 3
    # "a" is heard for frames 0-5; "b" and "c" only weakly at frames 6 and 7, then silence.
    logits = torch.full((12, 4), -6.0)
    logits[:6, a] = 0.0
    logits[6, b] = -0.5
    logits[7, c] = -0.5
    logits[8:, blank] = 0.0
    emission = torch.log_softmax(logits, dim=-1)
    groups = [[a], [b, c]]

    plain = align_with_source_priors(
        emission, groups, blank=blank, frame_shift_seconds=0.02, source_priors=[]
    )
    ones = align_with_minimum_durations(emission, groups, blank=blank, minimum_frames=[1, 1, 1])
    assert [(s.start, s.end) for g in ones for s in g] == [
        (s.start, s.end) for g in plain for s in g
    ]
    assert [(s.start, s.end) for s in plain[1]] == [(6, 7), (7, 8)]  # crammed: one frame each

    floored = align_with_minimum_durations(emission, groups, blank=blank, minimum_frames=[2, 2, 2])
    assert all(span.end - span.start >= 2 for group in floored for span in group)


def test_token_minimum_frames_give_extra_frames_to_vowels():
    from ctc_segmentation import token_minimum_frames

    vowels = {1}
    # One group of three letters with a vowel in the middle; floor of 2 per letter, 9 per group.
    assert token_minimum_frames([[5, 1, 6]], 2, 9, vowels) == [2, 5, 2]
    # No vowel: extra frames go to the first letter; long groups need no extra.
    assert token_minimum_frames([[5, 6], [5, 6, 7, 8, 9]], 1, 5, vowels) == [4, 1, 1, 1, 1, 1, 1]


def test_paced_minimum_frames_follow_each_lines_pace():
    from ctc_segmentation import paced_minimum_frames
    from torchaudio.functional import TokenSpan

    def spans(starts, length):
        return [[TokenSpan(1, s, s + length, 0.9)] for s in starts]

    groups = [[1]] * 6
    # Line 1: slow, a token every 15 frames. Line 2: fast rap, a token every 4 frames (80 ms),
    # plus one long pause that must not slow its pace down.
    plain = spans([0, 15, 30], 10) + spans([100, 104, 160], 4)
    minimums = paced_minimum_frames(
        groups,
        plain,
        [(0, 3), (3, 6)],
        ratio=0.9,
        letter_cap=2,
        group_cap=5,
        pause_frames=20,
        vowel_ids={1},
    )
    assert minimums[:3] == [5, 5, 5]  # capped at the full floor
    assert minimums[3:] == [3, 3, 3]  # 0.9 x 4-frame pace, under the real pace

    # A fast line of long words: letters must not get the 2-frame floor (12 letters in 12 frames).
    long_words = [[1, 2, 3, 4]] * 3
    fast = spans([0, 4, 8], 4)
    minimums = paced_minimum_frames(
        long_words,
        fast,
        [(0, 3)],
        ratio=0.9,
        letter_cap=2,
        group_cap=5,
        pause_frames=20,
        vowel_ids={1},
    )
    assert minimums == [1, 1, 1, 1] * 3


def test_vocal_activity_costs_only_letters_in_quiet_frames():
    from ctc_segmentation import vocal_activity, with_vocal_activity

    loud = torch.sin(torch.arange(320 * 10, dtype=torch.float32))  # 10 frames of signal
    waveform = torch.cat((torch.zeros(320 * 10), loud))
    activity = vocal_activity(waveform, 19)
    assert activity[:8].max() < 0.1 and activity[-5:].min() > 0.9

    emission = torch.log_softmax(torch.zeros(19, 3), dim=-1)
    adjusted = with_vocal_activity(emission, activity, blank=0, weight=2.0)
    assert torch.equal(adjusted[:, 0], emission[:, 0])  # blanks unchanged
    assert (adjusted[:8, 1] < emission[:8, 1] - 1.5).all()  # quiet: letters cost more
    assert torch.allclose(adjusted[-5:, 1], emission[-5:, 1], atol=0.2)  # singing: about free
