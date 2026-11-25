import torch

from worker.kara.aligner.ctc import align_ctc  # noqa: E402
from worker.kara.aligner.decoder import (  # noqa: E402
    _assign_ctc_blank_holds,
    _decode_background_blocks,
    _inference_chunk_starts,
    _place_inline_backing,
    with_stem_reading,
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

    spans = align_ctc(emission, [[1], [2]], blank=0)

    assert [(span.start, span.end) for group in spans for span in group] == [(1, 3), (4, 6)]


def test_repeated_ctc_labels_require_an_intervening_blank():
    emission = _emission([0, 1, 0, 1, 0])

    spans = align_ctc(emission, [[1], [1]], blank=0)

    assert spans[0][0].end <= spans[1][0].start


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
        align_ctc(emission, [[1], [2], [3]], blank=0)
    except RuntimeError as error:
        assert "fewer frames" in str(error)
    else:
        raise AssertionError("expected alignment failure")


def test_minimum_frames_counts_repeats_across_group_boundaries():
    from worker.kara.aligner.ctc import minimum_ctc_frames

    assert minimum_ctc_frames([]) == 0
    assert minimum_ctc_frames([[1, 2], [3]]) == 3
    assert minimum_ctc_frames([[1, 1], [1, 2]]) == 6


def test_repeated_targets_reject_windows_without_room_for_blanks():
    emission = torch.log_softmax(torch.zeros(2, 3), dim=-1)
    try:
        align_ctc(emission, [[1, 1]], blank=0)
    except RuntimeError as error:
        assert "fewer frames" in str(error)
    else:
        raise AssertionError("Repeated labels require an intervening blank frame")


def test_minimum_durations_match_plain_ctc_at_one_frame_and_stop_cramming():
    from worker.kara.aligner.ctc import align_with_minimum_durations

    blank, a, b, c = 0, 1, 2, 3
    # "a" is heard for frames 0-5; "b" and "c" only weakly at frames 6 and 7, then silence.
    logits = torch.full((12, 4), -6.0)
    logits[:6, a] = 0.0
    logits[6, b] = -0.5
    logits[7, c] = -0.5
    logits[8:, blank] = 0.0
    emission = torch.log_softmax(logits, dim=-1)
    groups = [[a], [b, c]]

    plain = align_ctc(emission, groups, blank=blank)
    ones = align_with_minimum_durations(emission, groups, blank=blank, minimum_frames=[1, 1, 1])
    assert [(s.start, s.end) for g in ones for s in g] == [
        (s.start, s.end) for g in plain for s in g
    ]
    assert [(s.start, s.end) for s in plain[1]] == [(6, 7), (7, 8)]  # crammed: one frame each

    floored = align_with_minimum_durations(emission, groups, blank=blank, minimum_frames=[2, 2, 2])
    assert all(span.end - span.start >= 2 for group in floored for span in group)


def test_token_minimum_frames_give_extra_frames_to_vowels():
    from worker.kara.aligner.ctc import token_minimum_frames

    vowels = {1}
    # One group of three letters with a vowel in the middle; floor of 2 per letter, 9 per group.
    assert token_minimum_frames([[5, 1, 6]], 2, 9, vowels) == [2, 5, 2]
    # No vowel: extra frames go to the first letter; long groups need no extra.
    assert token_minimum_frames([[5, 6], [5, 6, 7, 8, 9]], 1, 5, vowels) == [4, 1, 1, 1, 1, 1, 1]


def test_paced_minimum_frames_follow_each_lines_pace():
    from torchaudio.functional import TokenSpan

    from worker.kara.aligner.ctc import paced_minimum_frames

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
    from worker.kara.aligner.ctc import vocal_activity, with_vocal_activity

    loud = torch.sin(torch.arange(320 * 10, dtype=torch.float32))  # 10 frames of signal
    waveform = torch.cat((torch.zeros(320 * 10), loud))
    activity = vocal_activity(waveform, 19)
    assert activity[:8].max() < 0.1 and activity[-5:].min() > 0.9

    emission = torch.log_softmax(torch.zeros(19, 3), dim=-1)
    adjusted = with_vocal_activity(emission, activity, blank=0, weight=2.0)
    assert torch.equal(adjusted[:, 0], emission[:, 0])  # blanks unchanged
    assert (adjusted[:8, 1] < emission[:8, 1] - 1.5).all()  # quiet: letters cost more
    assert torch.allclose(adjusted[-5:, 1], emission[-5:, 1], atol=0.2)  # singing: about free


def test_vocal_activity_reads_loudness_at_the_true_frame_times():
    from worker.kara.aligner.ctc import vocal_activity

    # Silence, then signal from 0.5 s. Stitched chunks drop a frame at each seam, so a late frame's
    # true time runs ahead of index x 20 ms: here every frame is 100 ms (5 frames) further on.
    waveform = torch.cat((torch.zeros(8000), torch.sin(torch.arange(16000, dtype=torch.float32))))
    nominal = vocal_activity(waveform, 40)
    assert nominal[:24].max() < 0.1 and nominal[26:].min() > 0.9
    times = torch.arange(40, dtype=torch.float64) * 0.02 + 0.0125 + 0.1
    shifted = vocal_activity(waveform, 40, frame_times=times)
    assert shifted[:19].max() < 0.1 and shifted[21:].min() > 0.9
    # A clock of the wrong length is ignored, not misread.
    assert torch.equal(vocal_activity(waveform, 40, frame_times=times[:10]), nominal)


def test_optional_groups_leave_the_lead_path_only_when_they_cost_it_time():
    from worker.kara.aligner.ctc import align_with_minimum_durations

    blank, a, b, c, d = 0, 1, 2, 3, 4
    # Lead "a b" then "d" straight after; "(c)" is an echo sung over "d" (no frame hears c).
    emission = _emission([0, 1, 1, 0, 2, 2, 4, 4, 4, 0], vocabulary=5)
    groups = [[a], [b], [c], [d]]
    kwargs = {"blank": blank, "minimum_frames": [1, 1, 1, 1], "blank_free_tokens": True}

    forced = align_with_minimum_durations(emission, groups, **kwargs)
    skipped = align_with_minimum_durations(emission, groups, optional_groups={2}, **kwargs)
    assert skipped[2] == []  # left out
    assert [(s.start, s.end) for g in skipped for s in g] == [(1, 3), (4, 6), (6, 9)]
    assert (forced[1][0].end, forced[3][0].start) != (
        6,
        6,
    )  # walking through "c" takes b's or d's frames

    # Sung in sequence ("Lonely (so lonely)"), the echo stays on the lead path.
    sung = _emission([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4], vocabulary=5)
    kept = align_with_minimum_durations(sung, groups, optional_groups={2}, **kwargs)
    assert [(s.start, s.end) for s in kept[2]] == [(7, 9)]


def test_optional_runs_at_the_song_edges_and_without_options_change_nothing():
    from worker.kara.aligner.ctc import align_with_minimum_durations

    blank, a, b, c = 0, 1, 2, 3
    emission = _emission([0, 2, 2, 0, 0, 0], vocabulary=4)
    groups = [[a], [b], [c]]
    spans = align_with_minimum_durations(
        emission, groups, blank=blank, minimum_frames=[1, 1, 1], optional_groups={0, 2}
    )
    assert spans[0] == [] and spans[2] == [] and [(s.start, s.end) for s in spans[1]] == [(1, 3)]
    plain = align_with_minimum_durations(emission, groups, blank=blank, minimum_frames=[1, 1, 1])
    empty = align_with_minimum_durations(
        emission, groups, blank=blank, minimum_frames=[1, 1, 1], optional_groups=set()
    )
    assert [[(s.start, s.end) for s in g] for g in plain] == [
        [(s.start, s.end) for s in g] for g in empty
    ]


def _background_blocks(plan, encoded, lead, background, lead_emission=None, **options):
    """Background lines of ``plan`` decoded against lead tokens at ``lead`` (start, end) frames."""
    from types import SimpleNamespace

    from torchaudio.functional import TokenSpan

    model = SimpleNamespace(
        last_background_log_probs=background,
        blank=0,
        tokenizer=SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 2, "c": 3}),
    )
    lead_tokens = [i for _, flag, tokens in plan if not flag for i in tokens]
    spans = [
        [TokenSpan(encoded[i][0], start, end, 1.0)] for i, (start, end) in zip(lead_tokens, lead)
    ]
    ranges, offset = [], 0
    for _, flag, tokens in plan:
        if not flag:
            ranges.append((offset, offset + len(tokens)))
            offset += len(tokens)
    blocks = _decode_background_blocks(
        plan,
        encoded,
        spans,
        ranges,
        model,
        0.02,
        40,
        100,
        True,
        lead_emission=_heard(background.shape[0], []) if lead_emission is None else lead_emission,
        **options,
    )
    return [(group[0].start, group[-1].end) for block in blocks for group in block["spans"]]


def _heard(frames, sounds):
    """A head that hears silence except ``sounds``: (letter, start, end, strength)."""
    logits = torch.zeros(frames, 4)
    logits[:, 0] = 6.0
    for letter, start, end, strength in sounds:
        logits[start:end, 0] = 0.0
        logits[start:end, letter] = strength
    return torch.log_softmax(logits, dim=-1)


def test_backing_lines_the_lead_sings_keep_the_leads_time_between_lines():
    # The lead sings "a" then "b" 30 frames apart; the backing voice repeats both lines later. The
    # background head hears the first clearly and something like "b" right after it (a pickup).
    plan = [(0, False, [0]), (1, False, [1]), (2, True, [2]), (3, True, [3])]
    encoded = [[1], [2], [1], [2]]
    background = _heard(160, [(1, 70, 80, 8.0), (2, 82, 86, 8.0)])

    paced = _background_blocks(plan, encoded, [(10, 20), (40, 50)], background)

    assert paced[0][0] == 70
    # Not on the pickup 12 frames later: 0.9 of the lead's 30 frames between the lines.
    assert paced[1][0] - paced[0][0] >= 27


def test_backing_lines_after_the_lead_are_read_on_the_lead_head_too():
    # Nothing else is singing after the last lead line: the lead head hears the backing voice there.
    plan = [(0, False, [0]), (1, True, [1])]
    encoded = [[1], [2]]
    silent = _heard(200, [])
    lead_emission = _heard(200, [(1, 10, 20, 8.0), (2, 120, 130, 8.0)])

    spans = _background_blocks(plan, encoded, [(10, 20)], silent, lead_emission=lead_emission)

    assert spans == [(120, 130)]


def test_a_backing_block_does_not_start_on_top_of_the_lead_line_before_it():
    # The background head hears the long lead line too; the backing line is sung after it.
    plan = [(0, False, [0, 1]), (1, True, [2])]
    encoded = [[1], [1], [3]]
    background = _heard(200, [(3, 20, 26, 8.0), (3, 110, 120, 4.0)])

    paced = _background_blocks(plan, encoded, [(10, 20), (90, 100)], background)

    # Not at frame 20, where the head hears the lead line itself.
    assert paced[0][0] == 110


def test_the_text_head_places_a_backing_line_neither_head_can_read():
    plan = [(0, False, [0]), (1, True, [1])]
    encoded = [[1], [2]]
    silent = _heard(200, [])
    share = torch.full((200, 1), -6.0)
    share[150:160] = 0.0

    spans = _background_blocks(
        plan, encoded, [(10, 20)], silent, text_share=(share, {1: 0}), text_bonus=2.0
    )

    assert spans[0][0] == 150


def test_a_line_the_plain_pass_did_not_hear_takes_the_full_floor():
    from torchaudio.functional import TokenSpan

    from worker.kara.aligner.ctc import paced_minimum_frames

    # A hummed "Mm" the model reads as silence lands on one frame of the plain pass, and two
    # "Ooh, ooh" on a frame each: that is not a pace.
    plain = [[TokenSpan(1, 50, 51, 0.1)], [TokenSpan(2, 90, 91, 0.1)], [TokenSpan(2, 91, 92, 0.1)]]
    minimums = paced_minimum_frames(
        [[1], [2], [2]],
        plain,
        [(0, 1), (1, 3)],
        ratio=0.9,
        letter_cap=2,
        group_cap=5,
        pause_frames=20,
        vowel_ids={2},
    )
    assert minimums == [5, 5, 5]


def test_the_stem_reading_adds_letters_and_never_silences_the_mix():
    # Frames 0-9: only the mix hears "a" (the separator removed it). Frames 10-19: only the stem
    # hears "b" (instruments mask it). Frames 20-29: both are silent.
    mix = _heard(30, [(1, 0, 10, 8.0)])
    stem = _heard(30, [(2, 10, 20, 8.0)])

    fused = with_stem_reading(mix, stem, blank=0)

    assert torch.allclose(fused.exp().sum(dim=-1), torch.ones(30), atol=1e-5)
    assert (fused[:10].argmax(dim=-1) == 1).all()
    assert torch.allclose(fused[:10, 1], mix[:10, 1], atol=0.05)
    assert (fused[10:20].argmax(dim=-1) == 2).all()
    assert (fused[20:].argmax(dim=-1) == 0).all()


def test_a_stem_a_frame_shorter_than_the_mix_is_read_as_far_as_it_goes():
    mix = _heard(30, [(1, 0, 10, 8.0)])
    stem = _heard(28, [(2, 10, 20, 8.0)])

    fused = with_stem_reading(mix, stem, blank=0)

    assert fused.shape == mix.shape
    assert torch.allclose(fused[28:], mix[28:])


def test_the_text_head_places_an_inline_backing_word_the_background_head_cannot_read():
    from torchaudio.functional import TokenSpan

    # "a (b) a": the lead path squeezed the bracketed word between the lead words, and the
    # background head hears nothing; the text head's backing voice hears it at frames 60-70.
    def spans():
        return [
            [TokenSpan(1, 10, 20, 1.0)],
            [TokenSpan(2, 20, 22, 0.1)],
            [TokenSpan(1, 80, 90, 1.0)],
        ]

    encoded, flags, silent = [[1], [2], [1]], [False, True, False], _heard(120, [])
    share = torch.full((120, 1), -6.0)
    share[60:70] = 0.0

    plain = spans()
    _place_inline_backing(plain, encoded, flags, [(0, 3)], silent, 0, 0.02)
    placed = spans()
    moved = _place_inline_backing(
        placed,
        encoded,
        flags,
        [(0, 3)],
        silent,
        0,
        0.02,
        text_share=(share, {1: 0}),
        text_bonus=2.0,
    )

    assert moved == [1]
    assert not 60 <= plain[1][0].start < 70
    assert 60 <= placed[1][0].start < 70
    assert placed[1][0].token == 2
    assert [(g[0].start, g[-1].end) for g in (placed[0], placed[2])] == [(10, 20), (80, 90)]
