"""Global CTC forced alignment with soft source-line onset priors.

This module operates only on an already-computed emission lattice. It does not
load an acoustic model, crop audio, or infer line timings independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torchaudio.functional import TokenSpan


@dataclass(frozen=True)
class SourcePrior:
    target_index: int
    time_seconds: float
    weight: float = 1.0


def _huber(value: float, delta: float) -> float:
    value = abs(value)
    if value <= delta:
        return 0.5 * value * value / delta
    return value - 0.5 * delta


def minimum_ctc_frames(token_groups: list[list[int]]) -> int:
    """CTC needs one frame per token plus blanks between repeated labels."""
    targets = [token for group in token_groups for token in group]
    return len(targets) + sum(left == right for left, right in zip(targets, targets[1:]))


def align_with_source_priors(
    emission: torch.Tensor,
    token_groups: list[list[int]],
    *,
    blank: int,
    frame_shift_seconds: float,
    source_priors: list[SourcePrior],
    prior_weight: float = 0.8,
    prior_delta_seconds: float = 0.75,
) -> list[list[TokenSpan]]:
    """Align token groups globally using CTC Viterbi and soft onset priors.

    ``emission`` must contain log probabilities with shape ``[frames, labels]``.
    A prior is charged only when the path enters its target token. It therefore
    influences the line onset without pulling every token toward that onset.
    """
    if emission.ndim != 2 or emission.shape[0] == 0:
        raise ValueError("emission must have shape [frames, labels] and contain frames")
    targets = [token for group in token_groups for token in group]
    if not targets:
        raise ValueError("cannot align an empty token sequence")
    if emission.shape[0] < minimum_ctc_frames(token_groups):
        raise RuntimeError("CTC emission has fewer frames than target tokens")
    if blank < 0 or blank >= emission.shape[1]:
        raise ValueError("blank token is outside the emission vocabulary")
    if any(token < 0 or token >= emission.shape[1] for token in targets):
        raise ValueError("target token is outside the emission vocabulary")

    # Interleave blanks and labels. State 2*i+1 represents target i.
    state_labels = [blank]
    for token in targets:
        state_labels.extend((token, blank))
    labels = torch.tensor(state_labels, dtype=torch.long, device=emission.device)
    state_count = len(state_labels)
    frame_count = int(emission.shape[0])
    negative_inf = torch.tensor(float("-inf"), device=emission.device)

    prior_by_target = {
        prior.target_index: (prior.time_seconds, prior.weight) for prior in source_priors
    }
    if any(target < 0 or target >= len(targets) for target in prior_by_target):
        invalid = next(target for target in prior_by_target if target < 0 or target >= len(targets))
        raise ValueError(f"source prior target {invalid} is outside the transcript")
    prior_targets = list(prior_by_target)
    prior_states = torch.tensor(
        [target * 2 + 1 for target in prior_targets],
        dtype=torch.long,
        device=emission.device,
    )
    prior_times = torch.tensor(
        [prior_by_target[target][0] for target in prior_targets],
        dtype=emission.dtype,
        device=emission.device,
    )
    prior_weights = torch.tensor(
        [prior_by_target[target][1] for target in prior_targets],
        dtype=emission.dtype,
        device=emission.device,
    )
    scores = torch.full((state_count,), float("-inf"), device=emission.device)
    scores[0] = emission[0, blank]
    if targets:
        initial = emission[0, targets[0]]
        if 0 in prior_by_target:
            expected_time, weight = prior_by_target[0]
            initial -= prior_weight * weight * _huber(expected_time, prior_delta_seconds)
        scores[1] = initial

    # 0 means stay, 1 advance one state, 2 skip a blank. One byte per state.
    backpointers = torch.zeros(
        (frame_count, state_count), dtype=torch.uint8, device=emission.device
    )
    state_indexes = torch.arange(state_count, device=emission.device)
    is_label = state_indexes.remainder(2).eq(1)
    skip_allowed = torch.zeros(state_count, dtype=torch.bool, device=emission.device)
    if state_count > 2:
        skip_allowed[2:] = is_label[2:] & labels[2:].ne(labels[:-2])

    for frame in range(1, frame_count):
        stay = scores
        advance = torch.cat((negative_inf.reshape(1), scores[:-1]))
        skip = torch.cat((negative_inf.repeat(2), scores[:-2]))
        skip = torch.where(skip_allowed, skip, negative_inf)
        choices = torch.stack((stay, advance, skip), dim=0)

        # Priors belong to transitions into acoustic tokens. Compute all
        # penalties together; per-prior Python loops dominate long songs.
        if prior_states.numel():
            residual = (frame * frame_shift_seconds - prior_times).abs()
            huber = torch.where(
                residual <= prior_delta_seconds,
                0.5 * residual.square() / prior_delta_seconds,
                residual - 0.5 * prior_delta_seconds,
            )
            penalties = prior_weight * prior_weights * huber
            choices[1:, prior_states] -= penalties.unsqueeze(0)

        best_scores, best_moves = choices.max(dim=0)
        scores = best_scores + emission[frame].index_select(0, labels)
        backpointers[frame] = best_moves.to(dtype=torch.uint8)

    final_states = [state_count - 1, state_count - 2]
    state = max(final_states, key=lambda index: float(scores[index]))
    if not math.isfinite(float(scores[state])):
        raise RuntimeError("no finite CTC path exists for the supplied transcript")

    backpointers = backpointers.cpu()
    path = [state]
    for frame in range(frame_count - 1, 0, -1):
        move = int(backpointers[frame, state])
        state -= move
        path.append(state)
    path.reverse()

    flat_spans: list[TokenSpan] = []
    for target_index, token in enumerate(targets):
        token_state = target_index * 2 + 1
        frames = [frame for frame, path_state in enumerate(path) if path_state == token_state]
        if not frames:
            raise RuntimeError(f"CTC path omitted target token {target_index}")
        token_scores = emission[frames, token].exp()
        flat_spans.append(TokenSpan(token, frames[0], frames[-1] + 1, float(token_scores.mean())))

    grouped: list[list[TokenSpan]] = []
    offset = 0
    for group in token_groups:
        grouped.append(flat_spans[offset : offset + len(group)])
        offset += len(group)
    return grouped


def token_minimum_frames(
    token_groups: list[list[int]],
    letter_frames: int,
    group_frames: int,
    vowel_ids: set[int],
) -> list[int]:
    """Minimum frames per target: each letter at least ``letter_frames``, each acoustic token
    (group) at least ``group_frames``, its extra frames given to vowels (what singers hold)."""
    minimums: list[int] = []
    for group in token_groups:
        frames = [max(1, letter_frames)] * len(group)
        extra = group_frames - sum(frames)
        slots = [index for index, token in enumerate(group) if token in vowel_ids] or [0]
        step = 0
        while extra > 0 and frames:
            frames[slots[step % len(slots)]] += 1
            extra -= 1
            step += 1
        minimums.extend(frames)
    return minimums


def align_with_minimum_durations(
    emission: torch.Tensor,
    token_groups: list[list[int]],
    *,
    blank: int,
    minimum_frames: list[int],
) -> list[list[TokenSpan]]:
    """Global CTC Viterbi where target i occupies at least ``minimum_frames[i]`` frames.

    Plain CTC lets a letter take one frame, so with flat scores the best path crams words or holds
    one and squeezes the next. Each target becomes a chain of states walked one frame at a time
    before its last state may repeat; blanks are unchanged. All ones is the ordinary topology.
    """
    targets = [token for group in token_groups for token in group]
    if emission.ndim != 2 or emission.shape[0] == 0 or not targets:
        raise ValueError("emission and transcript must be non-empty")
    if len(minimum_frames) != len(targets) or any(frames < 1 for frames in minimum_frames):
        raise ValueError("every target needs a minimum of at least one frame")
    labels = [blank]
    kinds = [0]  # 0 blank, 1 first sub-state, 2 inner, 3 last, 4 first and last
    owners = [-1]
    for index, (token, frames) in enumerate(zip(targets, minimum_frames)):
        for step in range(frames):
            labels.append(token)
            kinds.append(4 if frames == 1 else 1 if step == 0 else 3 if step == frames - 1 else 2)
            owners.append(index)
        labels.append(blank)
        kinds.append(0)
        owners.append(-1)
    if emission.shape[0] < sum(minimum_frames):
        raise RuntimeError("CTC emission has fewer frames than the minimum durations need")
    device = emission.device
    state_labels = torch.tensor(labels, dtype=torch.long, device=device)
    state_kinds = torch.tensor(kinds, device=device)
    can_stay = (state_kinds == 0) | (state_kinds == 3) | (state_kinds == 4)
    can_skip = torch.zeros(len(labels), dtype=torch.bool, device=device)
    for state, kind in enumerate(kinds):
        if (
            kind in (1, 4)
            and state >= 2
            and kinds[state - 2] in (3, 4)
            and labels[state - 2] != labels[state]
        ):
            can_skip[state] = True
    negative_inf = torch.tensor(float("-inf"), device=device)
    scores = torch.full((len(labels),), float("-inf"), device=device)
    scores[0] = emission[0, blank]
    scores[1] = emission[0, labels[1]]
    backpointers = torch.zeros((emission.shape[0], len(labels)), dtype=torch.uint8, device=device)
    for frame in range(1, emission.shape[0]):
        stay = torch.where(can_stay, scores, negative_inf)
        advance = torch.cat((negative_inf.reshape(1), scores[:-1]))
        skip = torch.where(can_skip, torch.cat((negative_inf.repeat(2), scores[:-2])), negative_inf)
        best_scores, best_moves = torch.stack((stay, advance, skip)).max(dim=0)
        scores = best_scores + emission[frame].index_select(0, state_labels)
        backpointers[frame] = best_moves.to(dtype=torch.uint8)
    state = max((len(labels) - 1, len(labels) - 2), key=lambda index: float(scores[index]))
    if not math.isfinite(float(scores[state])):
        raise RuntimeError("no finite CTC path exists for the supplied transcript")
    backpointers = backpointers.cpu()
    path = [state]
    for frame in range(emission.shape[0] - 1, 0, -1):
        state -= int(backpointers[frame, state])
        path.append(state)
    path.reverse()
    frames_by_target: list[list[int]] = [[] for _ in targets]
    for frame, path_state in enumerate(path):
        if owners[path_state] >= 0:
            frames_by_target[owners[path_state]].append(frame)
    flat_spans = []
    for index, token in enumerate(targets):
        frames = frames_by_target[index]
        if not frames:
            raise RuntimeError(f"CTC path omitted target token {index}")
        token_scores = emission[frames, token].exp()
        flat_spans.append(TokenSpan(token, frames[0], frames[-1] + 1, float(token_scores.mean())))
    grouped: list[list[TokenSpan]] = []
    offset = 0
    for group in token_groups:
        grouped.append(flat_spans[offset : offset + len(group)])
        offset += len(group)
    return grouped


def paced_minimum_frames(
    token_groups: list[list[int]],
    plain_spans: list[list[TokenSpan]],
    line_ranges: list[tuple[int, int]],
    *,
    ratio: float,
    letter_cap: int,
    group_cap: int,
    pause_frames: int,
    vowel_ids: set[int],
) -> list[int]:
    """Minimum frames per target, scaled to each line's sung pace in a plain (unconstrained) pass.

    A fixed floor is right for most singing but longer than fast rap actually takes, which forces
    lines out of place. A line's pace is its span minus pauses longer than ``pause_frames`` between
    acoustic tokens: cramming can't shrink a span and pauses don't inflate it. Tokens get ``ratio``
    x the line's frames per token (at most ``group_cap``) and letters ``ratio`` x its frames per
    letter (at most ``letter_cap``), so fast rap keeps short syllables and one-frame letters.
    """
    minimums: list[int] = []
    for start, end in line_ranges:
        spans = [(group[0].start, group[-1].end) for group in plain_spans[start:end] if group]
        if not spans:
            continue
        spans.sort()
        total = max(span_end for _, span_end in spans) - spans[0][0]
        pauses = sum(
            max(0, later[0] - earlier[1])
            for earlier, later in zip(spans, spans[1:])
            if later[0] - earlier[1] > pause_frames
        )
        sung = max(1.0, total - pauses)
        letters = sum(len(group) for group in token_groups[start:end])
        group_floor = max(1, min(group_cap, int(ratio * sung / len(spans))))
        letter_floor = max(1, min(letter_cap, int(ratio * sung / max(1, letters))))
        minimums.extend(
            token_minimum_frames(token_groups[start:end], letter_floor, group_floor, vowel_ids)
        )
    return minimums


def vocal_activity(
    waveform: torch.Tensor, frame_count: int, hop: int = 320, window: int = 400
) -> torch.Tensor:
    """Per-frame singing activity in [0, 1] from a 16 kHz vocal stem.

    Loudness (dB) over each frame's receptive field, scaled between the song's 10th and 90th
    percentile, so quiet passages read near 0 and full singing near 1.
    """
    waveform = waveform.flatten().float()
    padded = torch.nn.functional.pad(
        waveform, (0, max(0, frame_count * hop + window - waveform.numel()))
    )
    frames = padded.unfold(0, window, hop)[:frame_count]
    loudness = 10 * torch.log10(frames.square().mean(dim=1) + 1e-10)
    low, high = torch.quantile(loudness, 0.1), torch.quantile(loudness, 0.9)
    return ((loudness - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)


def with_vocal_activity(
    emission: torch.Tensor, activity: torch.Tensor, blank: int, weight: float
) -> torch.Tensor:
    """Letters cost ``weight`` x (1 - activity) extra per frame, so the path can't hold a word
    through silence or start a line in an instrumental; blanks (pauses) are unchanged."""
    penalty = weight * (1 - activity.to(emission.device, emission.dtype))
    adjusted = emission - penalty.unsqueeze(1)
    adjusted[:, blank] = emission[:, blank]
    return adjusted
