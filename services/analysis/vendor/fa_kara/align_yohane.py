import logging
from abc import ABC, abstractmethod
import statistics
from collections.abc import Callable
from typing import cast

import math
import time

import torch
import torchaudio
from torchaudio.functional import (
    TokenSpan,
    equalizer_biquad,
    highpass_biquad,
    lowpass_biquad,
    merge_tokens,
    resample,
)
from torchaudio.pipelines import MMS_FA
from torchaudio.pipelines._wav2vec2 import aligner
from torchaudio.transforms import Fade
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from ctc_segmentation import (
    SourcePrior,
    align_with_minimum_durations,
    align_with_source_priors,
    minimum_ctc_frames,
    paced_minimum_frames,
    token_minimum_frames,
    vocal_activity,
    with_vocal_activity,
)

logger = logging.getLogger(__name__)

TokenizerFn = Callable[[list[str]], list[list[int]]]
_WAV2VEC_ALIGNER_CACHE: dict[tuple[str, bool], "Wav2Vec2ForcedAligner"] = {}
_MMS_ALIGNER_CACHE: "TorchAudioForcedAligner | None" = None


class ForcedAligner(ABC):
    @abstractmethod
    def tokenize(
        self,
        batch: list[str],
    ) -> list[list[int]]: ...

    @abstractmethod
    def align(
        self,
        tokens: list[list[int]],
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, list[list[TokenSpan]], int]: ...


class TorchAudioForcedAligner(ForcedAligner):
    """
    https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html
    """

    bundle = MMS_FA

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = self.bundle.get_tokenizer()
        self.model = self.bundle.get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pyright: ignore[reportPrivateImportUsage]
        self.model.to(self.device)
        self.aligner = self.bundle.get_aligner()

    def tokenize(self, batch: list[str]):
        return cast(list[list[int]], self.tokenizer(batch))

    def align(self, tokens: list[list[int]], waveform: torch.Tensor, sample_rate: int):
        logger.info(f"TorchAudioForcedAligner: running MMS_FA on {self.device}")
        waveform = resample(waveform, sample_rate, int(self.bundle.sample_rate))
        waveform = waveform.mean(0, keepdim=True)
        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))
            emission = cast(torch.Tensor, emission)
        token_spans = self.aligner(emission[0], tokens)
        return emission, token_spans, int(self.bundle.sample_rate)


def _inference_chunk_starts(
    total_samples: int, chunk_samples: int, overlap_samples: int
) -> list[int]:
    step_samples = chunk_samples - overlap_samples
    return [
        start
        for start in range(0, total_samples, step_samples)
        if start == 0 or total_samples - start > overlap_samples
    ]


class Wav2Vec2ForcedAligner(ForcedAligner):
    def __init__(self, model: str, *, use_gpu: bool = True) -> None:
        super().__init__()
        self.model_id = model
        self.processor = Wav2Vec2Processor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # pyright: ignore[reportArgumentType]
        blank = self.model.config.pad_token_id
        assert blank is not None
        self.blank = blank

    @property
    def tokenizer(self) -> Wav2Vec2CTCTokenizer:
        return self.processor.tokenizer  # pyright: ignore[reportAttributeAccessIssue]

    def tokenize(self, batch: list[str]):
        return [self.tokenizer.encode(e, add_special_tokens=False) for e in batch]

    def infer(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int, float]:
        """Compute one stitched full-track lattice from model-safe audio chunks."""
        logger.info("Wav2Vec2ForcedAligner: running %s on %s", self.model_id, self.device)
        target_sample_rate = self.processor.feature_extractor.sampling_rate  # pyright: ignore[reportAttributeAccessIssue]
        waveform = resample(waveform, sample_rate, target_sample_rate).mean(0)
        duration_seconds = waveform.numel() / target_sample_rate
        chunk_samples = 45 * target_sample_rate
        overlap_samples = 2 * target_sample_rate
        emissions = []
        starts = _inference_chunk_starts(waveform.numel(), chunk_samples, overlap_samples)
        for chunk_index, start in enumerate(starts):
            chunk = waveform[start : min(waveform.numel(), start + chunk_samples)]
            inputs = self.processor(
                audio=chunk.numpy(),
                sampling_rate=target_sample_rate,
                return_tensors="pt",  # pyright: ignore[reportCallIssue]
            )
            with torch.inference_mode():
                outputs = self.model(**inputs.to(self.device))
                local = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1).cpu()
            # Adjacent chunks share two seconds. Keep one half on each side so
            # no frame is decoded twice and edge-context damage is minimized.
            overlap_frames = round(local.shape[0] * overlap_samples / max(1, chunk.numel()))
            trim_left = overlap_frames // 2 if chunk_index else 0
            trim_right = (
                overlap_frames - overlap_frames // 2 if chunk_index + 1 < len(starts) else 0
            )
            emissions.append(local[trim_left : local.shape[0] - trim_right if trim_right else None])
        emission = torch.cat(emissions, dim=0)
        self.last_inference_passes = len(emissions)
        frame_shift_seconds = duration_seconds / emission.shape[0]
        return emission, target_sample_rate, frame_shift_seconds

    def align(self, tokens: list[list[int]], waveform: torch.Tensor, sample_rate: int):
        emission, sample_rate, _ = self.infer(waveform, sample_rate)
        token_spans = _align_token_spans(emission, tokens, blank=self.blank)
        return emission.unsqueeze(0), token_spans, sample_rate


def _align_token_spans(
    emission: torch.Tensor, tokens: list[list[int]], *, blank: int
) -> list[list[TokenSpan]]:
    aligned_tokens, scores = aligner._align_emission_and_tokens(
        emission, _flatten_token_sequences(tokens), blank=blank
    )
    spans = merge_tokens(aligned_tokens, scores, blank=blank)
    return _unflatten_token_spans(spans, [len(seq) for seq in tokens])


def _flatten_token_sequences(tokens: list[list[int]]) -> list[int]:
    return [token for seq in tokens for token in seq]


def _unflatten_token_spans(
    spans: list[TokenSpan], token_lengths: list[int]
) -> list[list[TokenSpan]]:
    if len(spans) != sum(token_lengths):
        raise RuntimeError(
            "Forced alignment returned a different number of token spans than tokens."
        )
    offset = 0
    grouped_spans: list[list[TokenSpan]] = []
    for length in token_lengths:
        grouped_spans.append(spans[offset : offset + length])
        offset += length
    return grouped_spans


def _scale_diagnostic_times(value, speed, key=None):
    """Return diagnostics in original-audio time after stretched inference."""
    if isinstance(value, dict):
        return {
            item_key: _scale_diagnostic_times(item, speed, item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_scale_diagnostic_times(item, speed, key) for item in value]
    if (
        key
        and key.endswith("_ms")
        and key != "source_drift_ms_per_minute"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return round(value * speed)
    return value


def _format_time(time_sec):
    minutes, remainder = divmod(max(0.0, time_sec), 60)
    seconds, centiseconds = divmod(remainder, 1)
    return f"[{int(minutes):02d}:{int(seconds):02d}:{math.floor(centiseconds * 100):02d}]"


def _block_ranges(line_count):
    """Partition lines into contextual blocks of three to five."""
    ranges = []
    start = 0
    while start < line_count:
        remaining = line_count - start
        if remaining <= 5:
            size = remaining
        else:
            size = 4
            if remaining - size < 3:
                size = remaining - 3
        ranges.append((start, start + size))
        start += size
    return ranges


def _span_result(token, spans, frame_duration, offset=0.0):
    if not spans:
        return {"token": token, "start": "[error]", "end": "[error]", "score": 0.0}
    start = offset + spans[0].start * frame_duration
    end = offset + spans[-1].end * frame_duration
    score = sum(float(span.score) for span in spans) / len(spans)
    return {
        "token": token,
        "start": _format_time(start),
        "end": _format_time(end),
        "original_start": start,
        "original_end": end,
        "score": score,
    }


def _candidate_cost(item, line_start, line_end, final_token, isolated):
    start = float(item.get("original_start", line_start))
    end = float(item.get("original_end", start))
    duration = max(0.0, end - start)
    score = max(0.001, float(item.get("score", 0.001)))
    cost = -math.log(score) * 0.35
    if duration < 0.04:
        cost += 0.4
    hold_limit = 2.5 if final_token else 1.2
    if duration > hold_limit:
        cost += (duration - hold_limit) * 1.5
    if start < line_start - 0.25:
        cost += (line_start - 0.25 - start) * 2.0
    if end > line_end + 0.25:
        cost += (end - line_end - 0.25) * 2.0
    if isolated and end >= line_end - 0.08:
        cost += 0.8
    return cost


def _hybrid_path(contextual, isolated, line_start, line_end):
    """Choose contextual or isolated timing per token with a monotonic DP."""
    if len(contextual) != len(isolated) or not contextual:
        return contextual
    candidates = [[contextual[index], isolated[index]] for index in range(len(contextual))]
    costs = [[float("inf"), float("inf")] for _ in candidates]
    previous = [[None, None] for _ in candidates]
    for state in range(2):
        costs[0][state] = _candidate_cost(
            candidates[0][state], line_start, line_end, len(candidates) == 1, state == 1
        )
    for index in range(1, len(candidates)):
        for state in range(2):
            current = candidates[index][state]
            current_start = float(current.get("original_start", 0))
            unary = _candidate_cost(
                current, line_start, line_end, index == len(candidates) - 1, state == 1
            )
            for prior_state in range(2):
                prior = candidates[index - 1][prior_state]
                prior_end = float(prior.get("original_end", 0))
                if current_start < prior_end - 0.02:
                    continue
                gap = max(0.0, current_start - prior_end)
                transition = max(0.0, gap - 1.0) * 0.4
                if state != prior_state:
                    transition += 0.12
                total = costs[index - 1][prior_state] + unary + transition
                if total < costs[index][state]:
                    costs[index][state] = total
                    previous[index][state] = prior_state
    state = min(range(2), key=lambda value: costs[-1][value])
    if math.isinf(costs[-1][state]):
        return contextual
    states = [state]
    for index in range(len(candidates) - 1, 0, -1):
        state = previous[index][state]
        if state is None:
            return contextual
        states.append(state)
    states.reverse()
    return [candidates[index][state] for index, state in enumerate(states)]


def _vocal_focus_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    focused = highpass_biquad(waveform.clone(), sample_rate, 100.0)
    focused = equalizer_biquad(focused, sample_rate, 250.0, -3.0, Q=0.8)
    focused = equalizer_biquad(focused, sample_rate, 2200.0, 4.0, Q=0.7)
    focused = equalizer_biquad(focused, sample_rate, 4200.0, 2.0, Q=1.0)
    focused = lowpass_biquad(focused, sample_rate, min(9500.0, sample_rate * 0.45))
    rms = focused.square().mean().sqrt().clamp_min(1e-6)
    return torch.tanh(focused * (0.12 / rms) * 1.8) / 1.8


def _line_span_score(groups: list[list[TokenSpan]]) -> float:
    scores = [float(span.score) for group in groups for span in group]
    return statistics.fmean(scores) if scores else 0.0


def _collapsed_token_count(
    groups: list[list[TokenSpan]], frame_shift: float, threshold_seconds: float = 0.12
) -> int:
    """Count acoustic tokens whose complete CTC span is implausibly short."""
    return sum(
        1
        for group in groups
        if group and (group[-1].end - group[0].start) * frame_shift < threshold_seconds
    )


def _assign_ctc_blank_holds(
    groups: list[list[TokenSpan]],
    line_ranges: list[tuple[int, int]],
    frame_duration: float = 0.02,
    max_extension_seconds: float = 0.2,
) -> list[list[TokenSpan]]:
    """Bounded display-only blank extension, not a silence detector."""
    import math

    if (
        not math.isfinite(max_extension_seconds)
        or max_extension_seconds < 0
        or not math.isfinite(frame_duration)
        or frame_duration <= 0
    ):
        raise ValueError("blank hold duration must be finite and nonnegative")
    max_frames = int(max_extension_seconds / frame_duration)
    result = [
        [TokenSpan(span.token, span.start, span.end, span.score) for span in group]
        for group in groups
    ]
    if not line_ranges:
        return result
    line_ends = {end - 1 for _, end in line_ranges if end > 0}
    for index in range(len(result) - 1):
        # Do not carry a final word through the pause before the next line.
        # Line containers handle display padding separately.
        if index in line_ends:
            continue
        current, following = result[index], result[index + 1]
        if not current or not following:
            continue
        next_start = following[0].start
        if next_start <= current[-1].end:
            continue
        # Blanks may also be silence; cap this display heuristic.
        last = current[-1]
        current[-1] = TokenSpan(
            last.token, last.start, min(next_start, last.end + max_frames), last.score
        )
    return result


def _display_span_results(tokens, spans, ranges, frame_duration, offset=0.0):
    """Token times in seconds. ``offset`` moves every time by a fixed amount (timing calibration)."""
    import os

    limit = float(os.environ.get("FA_KARA_MAX_BLANK_HOLD_SECONDS", "0.2"))
    display = _assign_ctc_blank_holds(spans, ranges, frame_duration, limit)
    results = []
    for token, raw, held in zip(tokens, spans, display, strict=True):
        item = _span_result(token, held, frame_duration, offset)
        acoustic = _span_result(token, raw, frame_duration, offset)
        item.update(
            acoustic_start=acoustic["original_start"], acoustic_end=acoustic["original_end"]
        )
        results.append(item)
    return results


def _maximum_internal_gap(groups: list[list[TokenSpan]], frame_shift: float) -> float:
    populated = [group for group in groups if group]
    return max(
        (
            (right[0].start - left[-1].end) * frame_shift
            for left, right in zip(populated, populated[1:])
        ),
        default=0.0,
    )


def _front_loaded_ratio(
    groups: list[list[TokenSpan]],
    frame_shift: float,
    interval_start: float,
    interval_end: float,
    early_fraction: float = 0.2,
) -> float:
    """Measure how much text finishes in the opening fraction of its source interval."""
    populated = [group for group in groups if group]
    interval_duration = interval_end - interval_start
    if len(populated) < 6 or interval_duration < 2.0:
        return 0.0
    cutoff = interval_start + interval_duration * early_fraction
    completed = sum(1 for group in populated if group[-1].end * frame_shift <= cutoff)
    return completed / len(populated)


def _is_collapsed_line(
    groups: list[list[TokenSpan]], frame_shift: float, interval_start: float, interval_end: float
) -> bool:
    """Identify strict fallback candidates without disturbing normal global CTC."""
    if not groups:
        return False
    if _maximum_internal_gap(groups, frame_shift) > 1.5:
        return True
    if _front_loaded_ratio(groups, frame_shift, interval_start, interval_end) >= 0.45:
        return True
    if _collapsed_token_count(groups, frame_shift) < 2:
        return False
    interval_duration = interval_end - interval_start
    if interval_duration < 0.5:
        return False
    occupied = (groups[-1][-1].end - groups[0][0].start) * frame_shift
    return occupied < interval_duration * 0.5


def _calibrate_source_priors(source_starts, observed_starts, target_indexes):
    """Fit source timestamps to acoustic anchors and reject local outliers."""
    if not (len(source_starts) == len(observed_starts) == len(target_indexes)):
        raise ValueError("source calibration inputs must have equal lengths")
    if not source_starts:
        return [], {"trusted_source_line_ratio": 0.0, "source_outlier_lines": []}
    slopes = []
    for left in range(len(source_starts)):
        for right in range(left + 1, len(source_starts)):
            source_delta = source_starts[right] - source_starts[left]
            if source_delta >= 5.0:
                slopes.append((observed_starts[right] - observed_starts[left]) / source_delta)
    slope = statistics.median(slopes) if slopes else 1.0
    # Gross slope estimates indicate bad acoustic anchors, not real playback
    # speed differences. Keep drift correction deliberately conservative.
    slope = min(1.02, max(0.98, slope))
    offset = statistics.median(
        observed - slope * source for source, observed in zip(source_starts, observed_starts)
    )
    signed_residuals = [
        observed - (slope * source + offset)
        for source, observed in zip(source_starts, observed_starts)
    ]
    median_residual = statistics.median(signed_residuals)
    mad = statistics.median(abs(value - median_residual) for value in signed_residuals)
    threshold = min(3.0, max(0.75, 3.0 * 1.4826 * mad))
    trusted = [abs(value - median_residual) <= threshold for value in signed_residuals]
    priors = [
        SourcePrior(target, max(0.0, slope * source + offset))
        for source, target, keep in zip(source_starts, target_indexes, trusted)
        if keep
    ]
    outliers = [index for index, keep in enumerate(trusted) if not keep]
    diagnostics = {
        "source_offset_ms": round(offset * 1000),
        "source_drift_ms_per_minute": round((slope - 1.0) * 60000),
        "median_source_residual_ms": round(
            statistics.median(abs(value) for value in signed_residuals) * 1000
        ),
        "source_outlier_lines": outliers,
        "trusted_source_line_ratio": sum(trusted) / len(trusted),
        "source_prior_residual_threshold_ms": round(threshold * 1000),
    }
    return priors, diagnostics


def align_audio_with_timeline_mms(
    audio_file_path, token_lines, line_starts_ms, sr=None, speed=1, use_gpu=True
):
    """Run Meta MMS_FA globally while preserving Echora's line structure."""
    del line_starts_ms, use_gpu
    if isinstance(audio_file_path, str):
        waveform, sample_rate = torchaudio.load(audio_file_path)
    else:
        waveform = torch.tensor(audio_file_path).float().unsqueeze(0)
        sample_rate = sr
    if sample_rate is None or speed <= 0:
        raise ValueError("sample rate and positive audio speed are required")
    acoustic_tokens = [token for line in token_lines for token in line if token]
    if not acoustic_tokens:
        raise ValueError("lyrics contain no alignable acoustic tokens")
    line_ranges = []
    offset = 0
    for line in token_lines:
        count = sum(1 for token in line if token)
        if count:
            line_ranges.append((offset, offset + count))
            offset += count
    global _MMS_ALIGNER_CACHE
    if _MMS_ALIGNER_CACHE is None:
        _MMS_ALIGNER_CACHE = TorchAudioForcedAligner()
    model = _MMS_ALIGNER_CACHE
    encoded = model.tokenize(acoustic_tokens)
    emission, spans, target_sample_rate = model.align(encoded, waveform, sample_rate)
    del target_sample_rate
    frame_shift = (waveform.shape[1] / sample_rate) / emission.shape[1]
    results = _display_span_results(acoustic_tokens, spans, line_ranges, frame_shift * speed)
    diagnostics = {
        "acoustic_aligner": "torchaudio_mms_fa",
        "ctc_blank_hold_assignment": "bounded_display_extension_not_silence_aware",
        "inference_audio_speed": speed,
        "line_count": len(line_ranges),
        "token_count": len(acoustic_tokens),
        "mean_ctc_score": statistics.fmean(item["score"] for item in results),
        "inference_passes": 1,
    }
    for result in results:
        result["inference_passes"] = 1
        result["source_diagnostics"] = diagnostics
    return results


def _wav2vec_aligner(model: str, use_gpu: bool) -> Wav2Vec2ForcedAligner:
    key = (model, bool(use_gpu))
    cached = _WAV2VEC_ALIGNER_CACHE.get(key)
    if cached is None:
        cached = Wav2Vec2ForcedAligner(model, use_gpu=use_gpu)
        _WAV2VEC_ALIGNER_CACHE[key] = cached
    return cached


def align_audio_with_timeline(
    audio_file_path,
    token_lines,
    line_starts_ms,
    sr=None,
    speed=1,
    use_gpu=True,
    hf_model_id=None,
    refine_all_lines=False,
    duration_aware_priors=False,
    reference_audio=None,
    min_letter_ms=0,
    min_token_ms=0,
    timing_offset_ms=0,
    duration_pace_ratio=0.0,
    vocal_activity_weight=0.0,
):
    """Align the complete song from one emission lattice with soft line priors.

    ``line_starts_ms=None`` aligns from the audio alone: the acoustic-only global path is the
    result, with no source calibration, priors, or source-windowed recovery.
    """
    if isinstance(audio_file_path, str):
        waveform, sample_rate = torchaudio.load(audio_file_path)
    else:
        waveform = torch.tensor(audio_file_path).float().unsqueeze(0)
        sample_rate = sr
    if sample_rate is None:
        raise ValueError("sample rate is required")
    if reference_audio is not None:
        reference_waveform = torch.tensor(reference_audio).float().unsqueeze(0)
        if reference_waveform.shape[1] != waveform.shape[1]:
            raise ValueError("reference audio duration must match alignment audio")
    else:
        reference_waveform = None
    if speed <= 0:
        raise ValueError("audio speed must be positive")
    if not token_lines or (line_starts_ms is not None and len(token_lines) != len(line_starts_ms)):
        raise ValueError("token lines and source timestamps must be non-empty and equal in length")

    acoustic_tokens = [token for line in token_lines for token in line if token]
    if not acoustic_tokens:
        raise ValueError("lyrics contain no alignable acoustic tokens")
    aligner_model = _wav2vec_aligner(hf_model_id, use_gpu)
    encoded = aligner_model.tokenize(acoustic_tokens)
    if len(encoded) != len(acoustic_tokens) or any(not group for group in encoded):
        raise RuntimeError("tokenizer omitted one or more acoustic tokens")

    emission, _, frame_shift = aligner_model.infer(waveform, sample_rate)
    original_inference_passes = aligner_model.last_inference_passes
    line_target_indexes = []
    line_acoustic_indexes = []
    line_acoustic_ranges = []
    acoustic_offset = 0
    target_offset = 0
    for line in token_lines:
        line_token_count = sum(1 for token in line if token)
        if line_token_count:
            line_target_indexes.append(target_offset)
            line_acoustic_indexes.append(acoustic_offset)
            line_acoustic_ranges.append((acoustic_offset, acoustic_offset + line_token_count))
            target_offset += sum(
                len(group)
                for group in encoded[acoustic_offset : acoustic_offset + line_token_count]
            )
            acoustic_offset += line_token_count

    # First pass is acoustic-only. It lets us estimate whether the supplied
    # source timeline has a global offset, drift, or isolated bad lines.
    initial_spans = align_with_source_priors(
        emission,
        encoded,
        blank=aligner_model.blank,
        frame_shift_seconds=frame_shift,
        source_priors=[],
    )
    if line_starts_ms is None:
        scoring_emission = emission
        if vocal_activity_weight > 0:
            # Decode on letters made costlier where the stem is quiet (token scores keep the
            # model's own probabilities).
            activity = vocal_activity(
                resample(waveform, sample_rate, 16000).mean(0), emission.shape[0]
            )
            emission = with_vocal_activity(
                emission, activity, aligner_model.blank, vocal_activity_weight
            )
            initial_spans = align_with_source_priors(
                emission,
                encoded,
                blank=aligner_model.blank,
                frame_shift_seconds=frame_shift,
                source_priors=[],
            )
        duration_floor = None
        if min_letter_ms > 0 or min_token_ms > 0:
            vocabulary = aligner_model.tokenizer.get_vocab()
            vowels = {vocabulary[letter] for letter in "aeiouy" if letter in vocabulary}
            letter_cap = max(1, round(min_letter_ms / 1000 / frame_shift))
            token_cap = round(min_token_ms / 1000 / frame_shift)
            if duration_pace_ratio > 0:
                # Scale each line's floor to its own pace in the plain pass (fast rap stays fast).
                minimums = paced_minimum_frames(
                    encoded,
                    initial_spans,
                    line_acoustic_ranges,
                    ratio=duration_pace_ratio,
                    letter_cap=letter_cap,
                    group_cap=max(1, token_cap),
                    pause_frames=round(0.4 / frame_shift),
                    vowel_ids=vowels,
                )
            else:
                minimums = token_minimum_frames(encoded, letter_cap, token_cap, vowels)
            try:
                initial_spans = align_with_minimum_durations(
                    emission,
                    encoded,
                    blank=aligner_model.blank,
                    minimum_frames=minimums,
                )
                duration_floor = {
                    "letter_ms": min_letter_ms,
                    "token_ms": min_token_ms,
                    "pace_ratio": duration_pace_ratio,
                }
            except RuntimeError:
                duration_floor = "too_dense_plain_path_kept"
        if len(initial_spans) != len(acoustic_tokens):
            raise RuntimeError("global CTC alignment returned the wrong token count")
        if scoring_emission is not emission:
            initial_spans = [
                [
                    TokenSpan(
                        span.token,
                        span.start,
                        span.end,
                        float(scoring_emission[span.start : span.end, span.token].exp().mean()),
                    )
                    for span in group
                ]
                for group in initial_spans
            ]
        diagnostics = {
            "source_timeline": "ignored",
            "minimum_duration": duration_floor,
            "timing_offset_ms": timing_offset_ms,
            "vocal_activity_weight": vocal_activity_weight,
            "ctc_blank_hold_assignment": "bounded_display_extension_not_silence_aware",
            "inference_audio_speed": speed,
        }
        # Frame k is reported at k * shift. Measured against reference word starts, those times
        # run early (frame centres, plus CTC firing ahead of the sung onset): calibrated offset.
        results = _display_span_results(
            acoustic_tokens,
            initial_spans,
            line_acoustic_ranges,
            frame_shift * speed,
            timing_offset_ms / 1000.0,
        )
        for result in results:
            result["inference_passes"] = original_inference_passes
            result["source_diagnostics"] = diagnostics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    observed_starts = [
        initial_spans[index][0].start * frame_shift for index in line_acoustic_indexes
    ]
    # The stretched waveform uses processed-audio coordinates. Source cues
    # arrive in original playback time and must be converted before decoding.
    source_starts = [max(0.0, float(value) / 1000.0 / speed) for value in line_starts_ms]
    priors, source_diagnostics = _calibrate_source_priors(
        source_starts, observed_starts, line_target_indexes
    )
    if duration_aware_priors:
        slope = 1.0 + float(source_diagnostics.get("source_drift_ms_per_minute", 0)) / 60000.0
        offset = float(source_diagnostics.get("source_offset_ms", 0)) / 1000.0
        internal_prior_count = 0
        for line_index, (acoustic_start, acoustic_end) in enumerate(line_acoustic_ranges[:-1]):
            group_count = acoustic_end - acoustic_start
            if group_count < 4:
                continue
            interval_start = max(0.0, slope * source_starts[line_index] + offset)
            interval_end = max(interval_start, slope * source_starts[line_index + 1] + offset)
            interval_duration = interval_end - interval_start
            if not 0.75 <= interval_duration <= 15.0:
                continue
            target_index = line_target_indexes[line_index]
            for group_offset in range(1, group_count):
                target_index += len(encoded[acoustic_start + group_offset - 1])
                fraction = group_offset / group_count
                # Inner priors are deliberately weak. The final acoustic token
                # gets more influence because front-loaded paths most often
                # truncate the sung ending there.
                weight = 0.32 if group_offset == group_count - 1 else 0.14
                priors.append(
                    SourcePrior(
                        target_index,
                        interval_start + interval_duration * fraction,
                        weight,
                    )
                )
                internal_prior_count += 1
        source_diagnostics["duration_aware_internal_priors"] = internal_prior_count
        source_diagnostics["duration_aware_prior_mode"] = "proportional_token_onsets"
    spans = align_with_source_priors(
        emission,
        encoded,
        blank=aligner_model.blank,
        frame_shift_seconds=frame_shift,
        source_priors=priors,
    )

    # Recover only consecutive outlier blocks. Isolated lines stay on the
    # global path, avoiding the aggressive per-line replacement experiments.
    outlier_lines = sorted(set(source_diagnostics.get("source_outlier_lines", [])))
    runs = []
    for line_index in outlier_lines:
        if runs and line_index == runs[-1][-1] + 1:
            runs[-1].append(line_index)
        else:
            runs.append([line_index])
    recovery_runs = []
    for run in runs:
        if len(run) >= 2:
            # Include the following line so a contested boundary is decoded
            # jointly instead of clipping the recovered block's final token.
            expanded = list(run)
            if expanded[-1] + 1 < len(line_acoustic_ranges):
                expanded.append(expanded[-1] + 1)
            recovery_runs.append(expanded)
        elif run[0] > 0:
            # A single bad line often means the boundary from its preceding
            # line was consumed too early. Retry both sides of that boundary.
            recovery_runs.append([run[0] - 1, run[0]])
        else:
            # A long instrumental intro can attract the first lyric line. It
            # has no preceding context, so retry it alone near its source cue.
            recovery_runs.append([0])
    recovered_lines = []
    recovery_candidates = []
    recovery_passes = 0
    if recovery_runs:
        focused_waveform = _vocal_focus_waveform(waveform, sample_rate)
        offset = float(source_diagnostics.get("source_offset_ms", 0)) / 1000.0
        slope = 1.0 + float(source_diagnostics.get("source_drift_ms_per_minute", 0)) / 60000.0
        duration = waveform.shape[1] / sample_rate
        for run in recovery_runs:
            first_line, last_line = run[0], run[-1]
            acoustic_start = line_acoustic_ranges[first_line][0]
            acoustic_end = line_acoustic_ranges[last_line][1]
            expanded_singleton = first_line not in outlier_lines and last_line in outlier_lines
            margin_before = 0.75 if expanded_singleton else 1.5
            margin_after = 0.25 if expanded_singleton else 1.5
            window_start = max(0.0, slope * source_starts[first_line] + offset - margin_before)
            if last_line + 1 < len(source_starts):
                window_end = min(
                    duration, slope * source_starts[last_line + 1] + offset + margin_after
                )
            else:
                window_end = min(duration, slope * source_starts[last_line] + offset + 8.0)
            if window_end <= window_start:
                continue
            segment = focused_waveform[
                :, int(window_start * sample_rate) : int(window_end * sample_rate)
            ]
            local_emission, _, local_shift = aligner_model.infer(segment, sample_rate)
            recovery_passes += aligner_model.last_inference_passes
            local_encoded = encoded[acoustic_start:acoustic_end]
            local_priors = []
            local_target = 0
            for line_index in run:
                local_priors.append(
                    SourcePrior(
                        local_target,
                        max(0.0, slope * source_starts[line_index] + offset - window_start),
                    )
                )
                start, end = line_acoustic_ranges[line_index]
                local_target += sum(len(group) for group in encoded[start:end])
            local_spans = align_with_source_priors(
                local_emission,
                local_encoded,
                blank=aligner_model.blank,
                frame_shift_seconds=local_shift,
                source_priors=local_priors,
                prior_weight=3.0 if expanded_singleton else 1.2,
            )
            local_start = window_start + local_spans[0][0].start * local_shift
            local_end = window_start + local_spans[-1][-1].end * local_shift
            previous_end = (
                spans[acoustic_start - 1][-1].end * frame_shift if acoustic_start else 0.0
            )
            next_start = (
                spans[acoustic_end][0].start * frame_shift
                if acoustic_end < len(spans)
                else float("inf")
            )
            local_score = _line_span_score(local_spans)
            original_score = _line_span_score(spans[acoustic_start:acoustic_end])
            candidate = {
                "lines": run,
                "start_ms": round(local_start * 1000),
                "end_ms": round(local_end * 1000),
                "previous_end_ms": round(previous_end * 1000),
                "next_start_ms": None
                if not math.isfinite(next_start)
                else round(next_start * 1000),
                "focused_score": local_score,
                "original_score": original_score,
            }
            previous_intrusion = max(0.0, previous_end - local_start)
            boundary_intrusion = max(0.0, local_end - next_start)
            maximum_intrusion = 1.25 if run == [0] else 0.35
            source_boundary = (
                slope * source_starts[last_line + 1] + offset
                if last_line + 1 < len(source_starts)
                else float("inf")
            )
            # A repeated phrase can make the global path place the following
            # line early too. If its source cue is well after that path and
            # the focused recovery finishes near the cue, clip this block at
            # the calibrated source boundary rather than rejecting it against
            # an already-invalid neighboring boundary.
            source_boundary_recovery = (
                math.isfinite(next_start)
                and math.isfinite(source_boundary)
                and (
                    abs(next_start - source_boundary) <= 0.75 or next_start < source_boundary - 1.0
                )
                and local_end <= source_boundary + 1.75
            )
            # Replacement cannot cross the unchanged next token. The source
            # cue may justify accepting a recovery, but clipping past this CTC
            # boundary would make the stored document non-monotonic.
            effective_boundary = next_start
            effective_intrusion = max(0.0, local_end - effective_boundary)
            recoverable_previous_intrusion = previous_intrusion <= maximum_intrusion and (
                source_boundary_recovery or acoustic_end == len(spans)
            )
            if (previous_intrusion > 0 and not recoverable_previous_intrusion) or (
                not source_boundary_recovery and boundary_intrusion > maximum_intrusion
            ):
                candidate["rejected"] = "adjacent_line_overlap"
                recovery_candidates.append(candidate)
                continue
            minimum_score_ratio = 0.6 if expanded_singleton else 0.7
            if local_score < original_score * minimum_score_ratio:
                candidate["rejected"] = "lower_ctc_score"
                recovery_candidates.append(candidate)
                continue
            if source_boundary_recovery:
                candidate["boundary_recovery"] = "calibrated_source_cue"
                candidate["source_boundary_ms"] = round(source_boundary * 1000)
            previous_frame = round(previous_end / frame_shift)
            boundary_frame = (
                round(effective_boundary / frame_shift)
                if math.isfinite(effective_boundary)
                else None
            )
            converted = []
            for group in local_spans:
                converted_group = []
                for span in group:
                    start_frame = max(
                        previous_frame,
                        round((window_start + span.start * local_shift) / frame_shift),
                    )
                    end_frame = max(
                        start_frame, round((window_start + span.end * local_shift) / frame_shift)
                    )
                    if boundary_frame is not None:
                        start_frame = min(start_frame, boundary_frame)
                        end_frame = min(max(start_frame, end_frame), boundary_frame)
                    converted_group.append(
                        TokenSpan(span.token, start_frame, end_frame, span.score)
                    )
                converted.append(converted_group)
            original_collapsed = _collapsed_token_count(
                spans[acoustic_start:acoustic_end],
                frame_shift,
            )
            converted_collapsed = _collapsed_token_count(converted, frame_shift)
            if converted_collapsed > original_collapsed:
                candidate["rejected"] = "boundary_clipping_collapsed_tokens"
                candidate["collapsed_tokens_before"] = original_collapsed
                candidate["collapsed_tokens_after_clipping"] = converted_collapsed
                recovery_candidates.append(candidate)
                continue
            candidate["accepted"] = True
            candidate["previous_boundary_intrusion_ms"] = round(previous_intrusion * 1000)
            candidate["boundary_intrusion_ms"] = round(effective_intrusion * 1000)
            recovery_candidates.append(candidate)
            spans[acoustic_start:acoustic_end] = converted
            recovered_lines.extend(run)
        source_diagnostics["vocal_focus_attempted_runs"] = recovery_runs
        source_diagnostics["vocal_focus_candidates"] = recovery_candidates
        source_diagnostics["vocal_focus_recovered_lines"] = recovered_lines
        source_diagnostics["vocal_focus_inference_passes"] = recovery_passes

    # Strict collapsed-token fallback. Global CTC (including Twilight's
    # consecutive-block recovery above) remains authoritative unless a line
    # has multiple sub-40ms tokens while using under half of its corrected
    # source interval. Retry that line and its follower independently, never
    # as a combined block, and replace only when collapse count decreases.
    offset = float(source_diagnostics.get("source_offset_ms", 0)) / 1000.0
    slope = 1.0 + float(source_diagnostics.get("source_drift_ms_per_minute", 0)) / 60000.0
    duration = waveform.shape[1] / sample_rate
    corrected_starts = [min(duration, max(0.0, slope * start + offset)) for start in source_starts]
    corrected_ends = corrected_starts[1:] + [duration]
    collapsed_lines = []
    for line_index, (start, end) in enumerate(line_acoustic_ranges):
        if _is_collapsed_line(
            spans[start:end], frame_shift, corrected_starts[line_index], corrected_ends[line_index]
        ):
            collapsed_lines.append(line_index)
    retry_lines = (
        list(range(len(line_acoustic_ranges)))
        if refine_all_lines
        else sorted(
            {
                candidate
                for line in collapsed_lines
                for candidate in (line, line + 1)
                if candidate < len(line_acoustic_ranges)
            }
        )
    )
    collapsed_candidates = []
    if retry_lines:
        focused_waveform = _vocal_focus_waveform(waveform, sample_rate)
        focused_reference_waveform = (
            _vocal_focus_waveform(reference_waveform, sample_rate)
            if reference_waveform is not None
            else None
        )
        for line_index in retry_lines:
            acoustic_start, acoustic_end = line_acoustic_ranges[line_index]
            window_start, window_end = corrected_starts[line_index], corrected_ends[line_index]
            candidate = {
                "line": line_index,
                "source_start_ms": round(window_start * 1000),
                "source_end_ms": round(window_end * 1000),
            }
            if window_end <= window_start:
                candidate["rejected"] = "empty_corrected_source_interval"
                collapsed_candidates.append(candidate)
                continue
            context_start = max(0, acoustic_start - 1) if refine_all_lines else acoustic_start
            context_end = min(len(encoded), acoustic_end + 1) if refine_all_lines else acoustic_end
            inference_window_start = (
                max(0.0, window_start - 0.75) if refine_all_lines else window_start
            )
            inference_window_end = (
                min(duration, window_end + 0.75) if refine_all_lines else window_end
            )
            segment = focused_waveform[
                :,
                int(inference_window_start * sample_rate) : int(inference_window_end * sample_rate),
            ]
            local_emission, _, local_shift = aligner_model.infer(segment, sample_rate)
            recovery_passes += aligner_model.last_inference_passes
            required_frames = minimum_ctc_frames(encoded[context_start:context_end])
            if local_emission.shape[0] < required_frames:
                candidate.update(
                    rejected="insufficient_ctc_frames",
                    available_frames=int(local_emission.shape[0]),
                    required_frames=required_frames,
                )
                collapsed_candidates.append(candidate)
                continue
            contextual_spans = align_with_source_priors(
                local_emission,
                encoded[context_start:context_end],
                blank=aligner_model.blank,
                frame_shift_seconds=local_shift,
                source_priors=[],
            )
            target_offset = acoustic_start - context_start
            local_spans = contextual_spans[
                target_offset : target_offset + acoustic_end - acoustic_start
            ]
            candidate_audio_source = "vocal_stem"
            if refine_all_lines and focused_reference_waveform is not None:
                reference_segment = focused_reference_waveform[
                    :,
                    int(inference_window_start * sample_rate) : int(
                        inference_window_end * sample_rate
                    ),
                ]
                reference_emission, _, reference_shift = aligner_model.infer(
                    reference_segment, sample_rate
                )
                recovery_passes += aligner_model.last_inference_passes
                if reference_emission.shape[0] >= required_frames:
                    reference_contextual_spans = align_with_source_priors(
                        reference_emission,
                        encoded[context_start:context_end],
                        blank=aligner_model.blank,
                        frame_shift_seconds=reference_shift,
                        source_priors=[],
                    )
                    reference_spans = reference_contextual_spans[
                        target_offset : target_offset + acoustic_end - acoustic_start
                    ]
                    # The mix can recover masked consonants or endings, but only
                    # replace the stem candidate when its acoustic confidence wins.
                    if _line_span_score(reference_spans) > _line_span_score(local_spans):
                        local_spans = reference_spans
                        local_shift = reference_shift
                        candidate_audio_source = "original_mix"
                else:
                    candidate["reference_rejected"] = "insufficient_ctc_frames"
            if refine_all_lines:
                candidate["context_tokens"] = {
                    "before": target_offset,
                    "after": context_end - acoustic_end,
                    "audio_margin_ms": 750,
                    "candidates": "vocal_stem+original_mix"
                    if focused_reference_waveform is not None
                    else "vocal_stem",
                }
                candidate["selected_audio_source"] = candidate_audio_source
            old_groups = spans[acoustic_start:acoustic_end]
            old_count = _collapsed_token_count(old_groups, frame_shift)
            new_count = _collapsed_token_count(local_spans, local_shift)
            old_gap = _maximum_internal_gap(old_groups, frame_shift)
            new_gap = _maximum_internal_gap(local_spans, local_shift)
            old_front_load = _front_loaded_ratio(
                old_groups,
                frame_shift,
                window_start,
                window_end,
            )
            old_score = _line_span_score(old_groups)
            new_score = _line_span_score(local_spans)
            new_front_load = _front_loaded_ratio(
                local_spans,
                local_shift,
                window_start - inference_window_start,
                window_end - inference_window_start,
            )
            local_start = inference_window_start + local_spans[0][0].start * local_shift
            local_end = inference_window_start + local_spans[-1][-1].end * local_shift
            previous_end = (
                spans[acoustic_start - 1][-1].end * frame_shift if acoustic_start else 0.0
            )
            next_start = (
                spans[acoustic_end][0].start * frame_shift
                if acoustic_end < len(spans)
                else duration
            )
            candidate.update(
                {
                    "collapsed_tokens_before": old_count,
                    "collapsed_tokens_after": new_count,
                    "internal_gap_before_ms": round(old_gap * 1000),
                    "internal_gap_after_ms": round(new_gap * 1000),
                    "front_loaded_ratio_before": old_front_load,
                    "front_loaded_ratio_after": new_front_load,
                    "ctc_score_before": old_score,
                    "ctc_score_after": new_score,
                    "start_ms": round(local_start * 1000),
                    "end_ms": round(local_end * 1000),
                }
            )
            if local_start < previous_end or local_end > next_start:
                candidate["rejected"] = "non_monotonic"
            elif refine_all_lines and new_score < old_score * 0.9:
                candidate["rejected"] = "lower_ctc_score"
            elif not (
                (
                    refine_all_lines
                    and new_score >= old_score * 0.9
                    and new_count <= old_count
                    and new_gap <= max(1.0, old_gap)
                    and new_front_load <= max(0.35, old_front_load + 0.1)
                )
                or new_count < old_count
                or (old_gap > 1.5 and new_gap < min(1.0, old_gap * 0.5))
                or (
                    old_front_load >= 0.45
                    and new_front_load <= min(0.35, old_front_load - 0.2)
                    and new_count <= old_count
                    and new_gap <= max(1.0, old_gap)
                )
            ):
                candidate["rejected"] = "collapse_gap_or_pacing_not_reduced"
            else:
                candidate["accepted"] = True
                spans[acoustic_start:acoustic_end] = [
                    [
                        TokenSpan(
                            span.token,
                            round(
                                (inference_window_start + span.start * local_shift) / frame_shift
                            ),
                            round((inference_window_start + span.end * local_shift) / frame_shift),
                            span.score,
                        )
                        for span in group
                    ]
                    for group in local_spans
                ]
            collapsed_candidates.append(candidate)
        source_diagnostics["line_refinement_mode"] = "all" if refine_all_lines else "anomalies"
        source_diagnostics["collapsed_fallback_trigger_lines"] = collapsed_lines
        source_diagnostics["collapsed_fallback_retry_lines"] = retry_lines
        source_diagnostics["collapsed_fallback_candidates"] = collapsed_candidates
        source_diagnostics["vocal_focus_inference_passes"] = recovery_passes

    if len(spans) != len(acoustic_tokens):
        raise RuntimeError("global CTC alignment returned the wrong token count")
    source_diagnostics["ctc_blank_hold_assignment"] = "bounded_display_extension_not_silence_aware"
    results = _display_span_results(
        acoustic_tokens, spans, line_acoustic_ranges, frame_shift * speed
    )
    total_inference_passes = original_inference_passes + recovery_passes
    source_diagnostics["inference_audio_speed"] = speed
    source_diagnostics = _scale_diagnostic_times(source_diagnostics, speed)
    for result in results:
        result["inference_passes"] = total_inference_passes
        result["source_diagnostics"] = source_diagnostics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def align_audio_with_text(
    audio_file_path,
    text_tokens,
    non_silent_ranges=[],
    sr=None,
    speed=1,
    use_gpu=True,
    hf_model_id=None,
):
    "Hugging Face 微调模型"

    start_time = time.time()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    try:
        if isinstance(audio_file_path, str):
            waveform, sample_rate = torchaudio.load(audio_file_path)
        else:
            waveform = torch.tensor(audio_file_path).float()
            waveform = waveform.unsqueeze(0)
            sample_rate = sr

        # 处理非静音区域
        if non_silent_ranges:
            # 将时间(秒)转换为样本点
            total_samples = waveform.shape[1]
            sample_ranges = []
            for start_sec, end_sec in non_silent_ranges:
                start_sample = int(start_sec * sample_rate / speed)
                end_sample = min(int(end_sec * sample_rate / speed), total_samples)
                sample_ranges.append((start_sample, end_sample))

            # 提取并拼接非静音片段
            segments = []
            for start, end in sample_ranges:
                segments.append(waveform[:, start:end])
            waveform = torch.cat(segments, dim=1)

        # 处理有效token
        valid_tokens = [token for token in text_tokens if token]

        # from yohane-2026.5.0
        torch_aligner = Wav2Vec2ForcedAligner(hf_model_id)
        tokens = torch_aligner.tokenize(valid_tokens)
        _, token_spans, tgt_sample_rate = torch_aligner.align(tokens, waveform, sample_rate)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 时间转换参数
        frame_duration = 1.0 / tgt_sample_rate * 320 * speed
        results = []

        # 映射回原始时间
        def map_to_original_time(adjusted_time):
            """将处理后的时间映射回原始音频时间"""
            if not non_silent_ranges:
                return adjusted_time

            cumulative_duration = 0.0
            for start_sec, end_sec in non_silent_ranges:
                segment_duration = end_sec - start_sec
                if adjusted_time < cumulative_duration + segment_duration:
                    return start_sec + (adjusted_time - cumulative_duration)
                cumulative_duration += segment_duration
            return non_silent_ranges[-1][1]  # 超出范围返回最后时间

        # 处理每个token的时间对齐
        for i, spans in enumerate(token_spans):
            if not spans:
                results.append({"token": valid_tokens[i], "start": "[error]", "end": "[error]"})
                continue

            # 获取调整后的时间
            adjusted_start = spans[0].start * frame_duration
            adjusted_end = spans[-1].end * frame_duration

            # 映射回原始音频时间
            original_start = map_to_original_time(adjusted_start)
            original_end = map_to_original_time(adjusted_end)

            results.append(
                {
                    "token": valid_tokens[i],
                    "start": _format_time(original_start),
                    "end": _format_time(original_end),
                    "original_start": original_start,
                    "original_end": original_end,
                }
            )

        end_time = time.time()
        print("Alignment inference executed in", round(end_time - start_time, 3), "seconds")
        return results

    except Exception as e:
        print(f"Error during alignment: {e}")
        return []
