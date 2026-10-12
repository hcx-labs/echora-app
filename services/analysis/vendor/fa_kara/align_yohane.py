import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import math
import time

import torch
import torchaudio
from torchaudio.functional import (
    TokenSpan,
    merge_tokens,
    resample,
)
from torchaudio.pipelines import MMS_FA
from torchaudio.pipelines._wav2vec2 import aligner
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from ctc_segmentation import (
    align_with_minimum_durations,
    align_with_source_priors,
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


def _format_time(time_sec):
    minutes, remainder = divmod(max(0.0, time_sec), 60)
    seconds, centiseconds = divmod(remainder, 1)
    return f"[{int(minutes):02d}:{int(seconds):02d}:{math.floor(centiseconds * 100):02d}]"


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
    sr=None,
    speed=1,
    use_gpu=True,
    hf_model_id=None,
    min_letter_ms=0,
    min_token_ms=0,
    timing_offset_ms=0,
    duration_pace_ratio=0.0,
    vocal_activity_weight=0.0,
):
    """Align the complete song from the audio alone, in one global CTC pass.

    ``token_lines`` gives the lyric lines (for line ranges and pacing); no source times are used.
    """
    if isinstance(audio_file_path, str):
        waveform, sample_rate = torchaudio.load(audio_file_path)
    else:
        waveform = torch.tensor(audio_file_path).float().unsqueeze(0)
        sample_rate = sr
    if sample_rate is None:
        raise ValueError("sample rate is required")
    if speed <= 0:
        raise ValueError("audio speed must be positive")
    if not token_lines:
        raise ValueError("token lines must be non-empty")

    acoustic_tokens = [token for line in token_lines for token in line if token]
    if not acoustic_tokens:
        raise ValueError("lyrics contain no alignable acoustic tokens")
    aligner_model = _wav2vec_aligner(hf_model_id, use_gpu)
    encoded = aligner_model.tokenize(acoustic_tokens)
    if len(encoded) != len(acoustic_tokens) or any(not group for group in encoded):
        raise RuntimeError("tokenizer omitted one or more acoustic tokens")

    emission, _, frame_shift = aligner_model.infer(waveform, sample_rate)
    original_inference_passes = aligner_model.last_inference_passes
    line_acoustic_ranges = []
    acoustic_offset = 0
    for line in token_lines:
        line_token_count = sum(1 for token in line if token)
        if line_token_count:
            line_acoustic_ranges.append((acoustic_offset, acoustic_offset + line_token_count))
            acoustic_offset += line_token_count

    # Plain pass; also the pace reference for the duration floor.
    initial_spans = align_with_source_priors(
        emission,
        encoded,
        blank=aligner_model.blank,
        frame_shift_seconds=frame_shift,
        source_priors=[],
    )
    scoring_emission = emission
    if vocal_activity_weight > 0:
        # Decode on letters made costlier where the stem is quiet (token scores keep the
        # model's own probabilities).
        activity = vocal_activity(resample(waveform, sample_rate, 16000).mean(0), emission.shape[0])
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
