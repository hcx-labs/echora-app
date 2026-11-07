import logging
import os
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
from transformers import (
    AutoConfig,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from ctc_segmentation import (
    align_ctc,
    align_with_minimum_durations,
    onset_class,
    paced_minimum_frames,
    refine_group_starts,
    stitched_frame_times,
    token_minimum_frames,
    vocal_activity,
    with_vocal_activity,
)

# Auxiliary heads shipped next to a w2v-BERT checkpoint (mrfakename/w2v-bert-2.0-music-lyrics-ctc,
# carried over by convert_sylla_to_fa_kara.py): ``vad`` is one vocal-activity logit per frame,
# ``f0`` one voicing logit plus 360 pitch bins of 20 cents (bin b at 10 * 2 ** ((20 b + 1997.38) / 1200) Hz).
AUX_HEADS_FILE = "aux_heads.safetensors"
F0_BINS = 360
VOCAL_ACTIVITY_SOURCES = ("energy", "vad", "external")

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
    """CTC emissions from a fine-tuned checkpoint. ``wav2vec2`` checkpoints (MMS, omniASR: every
    shipped model) take the waveform through ``Wav2Vec2Processor``; a ``wav2vec2-bert`` checkpoint
    (w2v-BERT 2.0 / Sylla-600M, picked from ``config.json``) takes stacked fbank frames through
    ``Wav2Vec2BertProcessor`` on the same 20 ms grid with a 279.5-sample receptive-field centre."""

    def __init__(
        self,
        model: str,
        *,
        use_gpu: bool = True,
        fusion: str = "off",
        fusion_model: str | None = None,
    ) -> None:
        super().__init__()
        self.model_id = model
        self.backbone = _backbone(model)
        if self.backbone == "wav2vec2-bert":
            self.processor = Wav2Vec2BertProcessor.from_pretrained(model)
            self.model = Wav2Vec2BertForCTC.from_pretrained(model)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model)
            self.model = Wav2Vec2ForCTC.from_pretrained(model)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # pyright: ignore[reportArgumentType]
        blank = self.model.config.pad_token_id
        assert blank is not None
        self.blank = blank
        # S1: a checkpoint trained with the edge (token-start) head carries ``edge_head`` in its
        # config; Wav2Vec2ForCTC ignores the extra weights, so they are loaded here by name.
        self.edge_head: torch.nn.Linear | None = None
        if getattr(self.model.config, "edge_head", False):
            self.edge_head = _load_edge_head(model, self.model.lm_head.in_features).to(self.device)
        # Label round v2: a checkpoint trained with --background-head-weight carries a second
        # CTC-vocabulary head for backing vocals; its stitched log-probabilities are kept as
        # ``last_background_log_probs`` for separate-path decoding of background lines (B0).
        self.background_head: torch.nn.Linear | None = None
        if getattr(self.model.config, "background_head", False):
            self.background_head = _load_edge_head(
                model,
                self.model.lm_head.in_features,
                "background_head",
                self.model.lm_head.out_features,
            ).to(self.device)
        self.last_background_log_probs: torch.Tensor | None = None
        # §5j text-conditioned head (``config.text_aligner``): loaded only when FA_KARA_TEXT_BONUS > 0 asks
        # the lead decode to use it; the stitched fused frame embeddings are then kept for it.
        self.text_aligner = None
        text_record = getattr(self.model.config, "text_aligner", None)
        if text_record and float(os.environ.get("FA_KARA_TEXT_BONUS", "0")) > 0:
            self.text_aligner = _load_text_aligner(model, dict(text_record)).to(self.device)
        self.last_hidden: torch.Tensor | None = None
        # §5f MuQ fusion: a checkpoint trained with the frozen MuQ branch carries ``fusion`` in
        # its config and ``fusion.*`` weights (projection + gate). Opt-in: ``fusion="on"`` loads
        # the branch, ``"gate0"`` runs the plain encoder (ablation), ``"off"`` refuses.
        self.fusion_adapter: torch.nn.Module | None = None
        self.fusion_branch = None
        self.fusion_mode = fusion
        self._fusion_layer = 0
        fusion_record = getattr(self.model.config, "fusion", None)
        if fusion_record:
            from muq_fusion import FUSION_MODES, MuQBranch, load_fusion_adapter, load_fusion_lora

            if fusion not in FUSION_MODES:
                raise ValueError(f"fusion must be one of {FUSION_MODES}")
            if fusion == "off":
                raise RuntimeError(
                    f"{model} is a MuQ fusion checkpoint; set fusion='on' (or 'gate0') to load it"
                )
            self.fusion_adapter = load_fusion_adapter(
                model, dict(fusion_record), self.model.lm_head.in_features
            ).to(self.device)
            self._fusion_layer = int(fusion_record["layer"])
            if fusion == "gate0":
                self.fusion_adapter.force_zero = True
            else:
                self.fusion_branch = MuQBranch(
                    fusion_model or fusion_record["model"], device=self.device
                )
                # A checkpoint trained with --fusion-lora-rank carries LoRA adapters for MuQ.
                if fusion_record.get("lora"):
                    load_fusion_lora(model, dict(fusion_record["lora"])).to(self.device).attach(
                        self.fusion_branch
                    )
        # Vocal-activity and F0 heads (inference features; ``vocal_activity_source="vad"``).
        self.aux_heads: torch.nn.ModuleDict | None = _load_aux_heads(
            model, self.model.lm_head.in_features
        )
        if self.aux_heads is not None:
            self.aux_heads.to(self.device)
        self.last_edge_logits: torch.Tensor | None = None
        self.last_vad_logits: torch.Tensor | None = None
        self.last_f0_logits: torch.Tensor | None = None
        self.last_frame_times: torch.Tensor | None = None

    def vad_activity(self, scaled: bool = True) -> torch.Tensor:
        """Per-frame singing activity from the VAD head on the last inference (CPU, [0, 1]).

        The head was trained on the parent encoder and is under-confident on Sylla's (mean
        probability ~0.4 inside sung words, ~0.05 in silence, AUC ~0.94 against Apple word
        timings on full mixes), so by default the probabilities are scaled between the song's 10th
        and 90th percentile exactly as ``ctc_segmentation.vocal_activity`` scales loudness; this
        keeps ``vocal_activity_weight`` on the same footing for both sources. ``scaled=False``
        returns the raw sigmoid."""
        if self.last_vad_logits is None:
            raise RuntimeError(
                "vocal_activity_source='vad' needs a checkpoint with aux_heads.safetensors"
            )
        probability = torch.sigmoid(self.last_vad_logits.float())
        if not scaled:
            return probability
        low, high = torch.quantile(probability, 0.1), torch.quantile(probability, 0.9)
        return ((probability - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)

    def f0_track(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(voicing probability, pitch in Hz of the best bin) per frame from the F0 head, or None."""
        if self.last_f0_logits is None:
            return None
        return pitch_from_f0_logits(self.last_f0_logits)

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
        # Shared context between adjacent 45 s chunks (half kept on each side). 2 s is the shipped
        # value; FA_KARA_CHUNK_OVERLAP_SECONDS (B, opt-in) widens it so edge frames see more context.
        overlap_seconds = int(os.environ.get("FA_KARA_CHUNK_OVERLAP_SECONDS", "2"))
        overlap_samples = overlap_seconds * target_sample_rate
        emissions = []
        edge_parts: list[torch.Tensor] = []
        vad_parts: list[torch.Tensor] = []
        f0_parts: list[torch.Tensor] = []
        background_parts: list[torch.Tensor] = []
        hidden_parts: list[torch.Tensor] = []
        captured: dict[str, torch.Tensor] = {}
        current: dict[str, torch.Tensor] = {}

        def before_lm_head(_, args):
            hidden = args[0]
            if self.fusion_adapter is not None:
                hidden = self._fused_hidden(hidden, current["chunk"])
            captured["hidden"] = hidden
            return (hidden, *args[1:]) if self.fusion_adapter is not None else None

        hook = (
            self.model.lm_head.register_forward_pre_hook(before_lm_head)
            if self.edge_head is not None
            or self.background_head is not None
            or self.aux_heads is not None
            or self.fusion_adapter is not None
            or self.text_aligner is not None
            else None
        )
        starts = _inference_chunk_starts(waveform.numel(), chunk_samples, overlap_samples)
        try:
            for chunk_index, start in enumerate(starts):
                chunk = waveform[start : min(waveform.numel(), start + chunk_samples)]
                current["chunk"] = chunk
                inputs = self.processor(
                    audio=chunk.numpy(),
                    sampling_rate=target_sample_rate,
                    return_tensors="pt",  # pyright: ignore[reportCallIssue]
                )
                # w2v-BERT: fbank pairs are padded to a multiple of two; the attention mask
                # marks the valid model frames and the last one may be padding.
                valid = (
                    int(inputs["attention_mask"].sum())
                    if self.backbone == "wav2vec2-bert"
                    else None
                )
                with torch.inference_mode():
                    outputs = self.model(**inputs.to(self.device))
                    local = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1).cpu()
                    hidden = captured.get("hidden")
                    edge_local = (
                        self.edge_head(hidden)[0, :, 0].float().cpu()
                        if self.edge_head is not None and hidden is not None
                        else None
                    )
                    hidden_local = (
                        hidden[0].float().cpu()
                        if self.text_aligner is not None and hidden is not None
                        else None
                    )
                    background_local = (
                        torch.nn.functional.log_softmax(
                            self.background_head(hidden)[0].float(), dim=-1
                        ).cpu()
                        if self.background_head is not None and hidden is not None
                        else None
                    )
                    vad_local = f0_local = None
                    if self.aux_heads is not None and hidden is not None:
                        vad_local = self.aux_heads["vad"](hidden)[0, :, 0].float().cpu()
                        f0_local = self.aux_heads["f0"](hidden)[0].float().cpu()
                if valid is not None:
                    local = local[:valid]
                    edge_local = edge_local[:valid] if edge_local is not None else None
                    background_local = (
                        background_local[:valid] if background_local is not None else None
                    )
                    hidden_local = hidden_local[:valid] if hidden_local is not None else None
                    vad_local = vad_local[:valid] if vad_local is not None else None
                    f0_local = f0_local[:valid] if f0_local is not None else None
                # Adjacent chunks share two seconds. Keep one half on each side so
                # no frame is decoded twice and edge-context damage is minimized.
                overlap_frames = round(local.shape[0] * overlap_samples / max(1, chunk.numel()))
                trim_left = overlap_frames // 2 if chunk_index else 0
                trim_right = (
                    overlap_frames - overlap_frames // 2 if chunk_index + 1 < len(starts) else 0
                )
                stop = local.shape[0] - trim_right if trim_right else None
                emissions.append(local[trim_left:stop])
                if edge_local is not None:
                    edge_parts.append(edge_local[trim_left:stop])
                if vad_local is not None:
                    vad_parts.append(vad_local[trim_left:stop])
                if background_local is not None:
                    background_parts.append(background_local[trim_left:stop])
                if hidden_local is not None:
                    hidden_parts.append(hidden_local[trim_left:stop])
                if f0_local is not None:
                    f0_parts.append(f0_local[trim_left:stop])
        finally:
            if hook is not None:
                hook.remove()
        emission = torch.cat(emissions, dim=0)
        self.last_inference_passes = len(emissions)
        self.last_edge_logits = torch.cat(edge_parts) if edge_parts else None
        self.last_vad_logits = torch.cat(vad_parts) if vad_parts else None
        self.last_background_log_probs = torch.cat(background_parts) if background_parts else None
        self.last_hidden = torch.cat(hidden_parts) if hidden_parts else None
        self.last_f0_logits = torch.cat(f0_parts) if f0_parts else None
        # E3: the true receptive-field-centre time of every stitched frame (opt-in clock).
        self.last_frame_times = stitched_frame_times(
            waveform.numel(),
            sample_rate=target_sample_rate,
            backbone=self.backbone,
            overlap_seconds=overlap_seconds,
        )
        if self.last_frame_times.shape[0] != emission.shape[0]:
            logger.warning(
                "exact frame clock mismatch (%d times for %d frames); legacy clock only",
                self.last_frame_times.shape[0],
                emission.shape[0],
            )
            self.last_frame_times = None
        frame_shift_seconds = duration_seconds / emission.shape[0]
        return emission, target_sample_rate, frame_shift_seconds

    def text_token_share(
        self, emission, token_letters, voices, lines, window=None, overlap=None, backing=None
    ):
        """§5j: log P(token | singing) of the LEAD tokens [frames, lead tokens] over the last inference.

        The head reads the stitched fused frame embeddings (``last_hidden``) and the encoder's letter
        log-probabilities (``emission``) in windows of ``window`` frames sharing ``overlap`` (half kept on
        each side; defaults: ``text_aligner.frame_windows``, the length the head was trained on) and the
        whole song's tokens: ``token_letters`` (letter ids per token, text order), ``voices`` (0 lead /
        1 backing) and ``lines`` (line index per token).

        ``backing``: indexes of backing tokens; then returns ``(lead share, backing share)``, the second
        being the backing voice's log P(token | singing) of those tokens [frames, len(backing)]."""
        from text_aligner import (
            TRAINING_WINDOW_FRAMES,
            WINDOW_OVERLAP_FRAMES,
            collate_text,
            frame_windows,
        )
        from song_text import lyric_tokens, text_features

        window = TRAINING_WINDOW_FRAMES if window is None else window
        overlap = WINDOW_OVERLAP_FRAMES if overlap is None else overlap
        head = self.text_aligner
        version = getattr(self.model.config, "song_text_version", 1)
        tokens = lyric_tokens(token_letters, voices, lines, version)
        features = text_features(tokens, head.config.max_letters, head.config.letters)
        # Columns must stay in input order for the acoustic decoder's lead indexes.
        if [t["start"] for t, _ in tokens] != list(range(len(token_letters))):
            raise ValueError("Text lyrics must be in line/occurrence order")
        packed = {k: torch.tensor(v, dtype=torch.long) for k, v in features.items()}
        text = {k: v.to(self.device) for k, v in collate_text([packed], head.config).items()}
        hidden, frames = self.last_hidden, emission.shape[0]
        if hidden is None or hidden.shape[0] != frames:
            raise RuntimeError(
                "text-aligner head: stitched embeddings do not match the emission lattice"
            )
        lead_columns = torch.tensor([j for j, v in enumerate(voices) if v == 0], device=self.device)
        if backing and any(voices[j] != 1 for j in backing):
            raise ValueError("backing share: not a backing token")
        backing_voice = torch.tensor(
            [j for j, v in enumerate(voices) if v == 1], device=self.device
        )
        backing_columns = torch.tensor(backing or [], dtype=torch.long, device=self.device)
        parts, backing_parts = [], []
        with torch.inference_mode():
            for piece, keep in frame_windows(frames, window, overlap):
                h = hidden[piece].to(self.device)[None]
                letters = emission[piece].to(self.device, torch.float32)[None]
                padding = torch.zeros(h.shape[:2], dtype=torch.bool, device=self.device)
                out, back = head.forward_parts(
                    h,
                    padding,
                    text["letters"],
                    text["voice"],
                    text["text_padding"],
                    letters,
                    text["line"],
                    text["position"],
                )

                def voice_share(part, columns, voice_columns):
                    if "token" in part:
                        return part["token"][0][:, columns]
                    # an unfactorized head: the voice's columns renormalised without "none"
                    full = torch.full_like(part["combined"][0][:, 1:], float("-inf"))
                    full[:, voice_columns] = torch.log_softmax(
                        part["combined"][0][:, 1:][:, voice_columns], -1
                    )
                    return full[:, columns]

                parts.append(voice_share(out, lead_columns, lead_columns)[keep].float().cpu())
                if backing:
                    backing_parts.append(
                        voice_share(back, backing_columns, backing_voice)[keep].float().cpu()
                    )
        result = torch.cat(parts)
        if result.shape[0] != frames:
            raise RuntimeError(
                "text-aligner head: stitched output does not match the emission lattice"
            )
        if backing:
            return result, torch.cat(backing_parts)
        return result

    def _fused_hidden(self, hidden: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        """The encoder's last hidden state of one chunk plus the gated MuQ projection, sampled
        on this backbone's frame clock (the same arithmetic as the trainer's ``fusion_hidden``)."""
        from ctc_segmentation import receptive_field_centre_samples
        from muq_fusion import fusion_term

        assert self.fusion_adapter is not None
        rate = self.processor.feature_extractor.sampling_rate  # pyright: ignore[reportAttributeAccessIssue]
        term = fusion_term(
            self.fusion_adapter,
            self.fusion_branch,
            chunk[None].to(self.device),
            torch.tensor([chunk.numel()]),
            hidden.shape[1],
            self._fusion_layer,
            320 / rate,
            receptive_field_centre_samples(self.backbone) / rate,
        )
        return hidden if term is None else hidden + term.to(hidden.dtype)

    def align(self, tokens: list[list[int]], waveform: torch.Tensor, sample_rate: int):
        emission, sample_rate, _ = self.infer(waveform, sample_rate)
        token_spans = _align_token_spans(emission, tokens, blank=self.blank)
        return emission.unsqueeze(0), token_spans, sample_rate


def _backbone(model: str) -> str:
    """``wav2vec2`` (MMS/omniASR, the default path) or ``wav2vec2-bert`` from the checkpoint config."""
    model_type = AutoConfig.from_pretrained(model).model_type
    if model_type == "wav2vec2-bert":
        return "wav2vec2-bert"
    if model_type == "wav2vec2":
        return "wav2vec2"
    raise RuntimeError(f"unsupported forced-aligner backbone {model_type!r} in {model}")


def _load_aux_heads(model: str, in_features: int) -> torch.nn.ModuleDict | None:
    """``vad`` and ``f0`` linear heads from ``aux_heads.safetensors`` next to the checkpoint
    (local directory or hub repo), or None when the checkpoint ships none."""
    from transformers.utils import cached_file

    try:
        path = cached_file(model, AUX_HEADS_FILE, _raise_exceptions_for_missing_entries=False)
    except OSError:
        return None
    if not path:
        return None
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        state = {k: handle.get_tensor(k) for k in handle.keys()}
    expected = {"vad.weight", "vad.bias", "f0.weight", "f0.bias"}
    if set(state) != expected or state["vad.weight"].shape != (1, in_features):
        raise RuntimeError(f"unexpected auxiliary head layout in {path}: {sorted(state)}")
    heads = torch.nn.ModuleDict(
        {
            "vad": torch.nn.Linear(in_features, 1),
            "f0": torch.nn.Linear(in_features, int(state["f0.weight"].shape[0])),
        }
    )
    heads.load_state_dict(state)
    return heads.eval()


def pitch_from_f0_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(voicing probability, Hz of the most likely bin) per frame from the F0 head's
    ``1 + F0_BINS`` outputs: index 0 is the voicing logit, bin b (1-based) sits at
    10 * 2 ** ((20 b + 1997.38) / 1200) Hz (20-cent bins, about 32 Hz to 2 kHz)."""
    logits = logits.float()
    voicing = torch.sigmoid(logits[:, 0])
    bins = logits[:, 1:].argmax(dim=-1) + 1
    hertz = 10.0 * torch.pow(2.0, (20.0 * bins.float() + 1997.38) / 1200.0)
    return voicing, hertz


def _load_text_aligner(model_dir: str, record: dict):
    """§5j text-conditioned head (``text_aligner.*`` in model.safetensors, config ``text_aligner``)."""
    from safetensors import safe_open
    from text_aligner import TextAligner, TextAlignerConfig

    head = TextAligner(TextAlignerConfig(**record))
    with safe_open(
        os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        state = {
            k.removeprefix("text_aligner."): handle.get_tensor(k)
            for k in handle.keys()
            if k.startswith("text_aligner.")
        }
    head.load_state_dict(state)
    return head.eval()


def _load_edge_head(
    model_dir: str, in_features: int, prefix: str = "edge_head", outputs: int = 1
) -> torch.nn.Linear:
    """The S1 edge head (``edge_head.weight`` / ``edge_head.bias``) from a checkpoint directory;
    ``prefix``/``outputs`` load another linear head the same way (the v2 ``background_head``)."""
    import os

    head = torch.nn.Linear(in_features, outputs)
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    state: dict[str, torch.Tensor] = {}
    if os.path.isfile(safetensors_path):
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            state = {k: handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix + ".")}
    else:
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            full = torch.load(bin_path, map_location="cpu")
            state = {k: v for k, v in full.items() if k.startswith(prefix + ".")}
    if set(state) != {f"{prefix}.weight", f"{prefix}.bias"}:
        raise RuntimeError(
            f"{prefix} declared in config but its weights are missing in {model_dir}"
        )
    head.load_state_dict({k.removeprefix(prefix + "."): v for k, v in state.items()})
    return head.eval()


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


def _span_result(token, spans, frame_duration, offset=0.0, frame_times=None, fraction=0.0):
    """Token times in seconds. Legacy clock: frame k at k x ``frame_duration``. With
    ``frame_times`` (E3) frame k is at its own receptive-field centre, and ``fraction`` (S2)
    moves the start by a fraction of a frame."""
    if not spans:
        return {"token": token, "start": "[error]", "end": "[error]", "score": 0.0}
    if frame_times is None:
        start = offset + (spans[0].start + fraction) * frame_duration
        end = offset + spans[-1].end * frame_duration
    else:
        start = (
            offset
            + _frame_time(frame_times, spans[0].start, frame_duration)
            + fraction * frame_duration
        )
        end = offset + _frame_time(frame_times, spans[-1].end, frame_duration)
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
    second_stream: frozenset[int] | set[int] = frozenset(),
) -> list[list[TokenSpan]]:
    """Bounded display-only blank extension, not a silence detector.

    ``second_stream`` groups (optional brackets placed on their own) are timed apart from the
    lead: a word is only held up to the next word of its own stream on the same line."""
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
    line_of = {
        group: line for line, (start, end) in enumerate(line_ranges) for group in range(start, end)
    }
    for index in range(len(result) - 1):
        # Do not carry a final word through the pause before the next line.
        # Line containers handle display padding separately.
        if index in line_ends:
            continue
        following_index = index + 1
        if second_stream:
            mine = index in second_stream
            following_index = next(
                (k for k in range(index + 1, len(result)) if (k in second_stream) == mine), None
            )
            if following_index is None or line_of.get(following_index) != line_of.get(index):
                continue
        current, following = result[index], result[following_index]
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


def _frame_time(frame_times, frame, frame_duration):
    """Centre time of ``frame`` on the exact clock; one past the last frame extrapolates."""
    count = int(frame_times.shape[0])
    if frame < count:
        return float(frame_times[frame])
    return float(frame_times[count - 1]) + (frame - count + 1) * frame_duration


def _display_span_results(
    tokens,
    spans,
    ranges,
    frame_duration,
    offset=0.0,
    frame_times=None,
    fractions=None,
    token_offsets=None,
    second_stream=frozenset(),
):
    """Token times in seconds. ``offset`` moves every time by a fixed amount (timing calibration);
    ``token_offsets`` (S2 onset-class table) adds a per-token amount, ``fractions`` a sub-frame
    start per token, ``frame_times`` the exact chunk clock (E3)."""
    import os

    limit = float(os.environ.get("FA_KARA_MAX_BLANK_HOLD_SECONDS", "0.2"))
    display = _assign_ctc_blank_holds(spans, ranges, frame_duration, limit, second_stream)
    results = []
    for index, (token, raw, held) in enumerate(zip(tokens, spans, display, strict=True)):
        shift = offset + (token_offsets[index] if token_offsets is not None else 0.0)
        fraction = fractions[index] if fractions is not None else 0.0
        item = _span_result(token, held, frame_duration, shift, frame_times, fraction)
        acoustic = _span_result(token, raw, frame_duration, shift, frame_times, fraction)
        item.update(
            acoustic_start=acoustic["original_start"], acoustic_end=acoustic["original_end"]
        )
        results.append(item)
    return results


def _decode_background_blocks(
    plan,
    all_encoded,
    lead_spans,
    lead_ranges,
    aligner_model,
    frame_shift,
    min_letter_ms,
    min_token_ms,
    blank_free_tokens,
    margin_seconds=1.0,
):
    """B0: decode each block of consecutive background lines on the background head.

    A block's window runs from the start of the lead line before it to the end of the lead line
    after it (plus ``margin_seconds``), so an echo may sit on, or just after, the lead it answers.
    Inside the window the block is one duration-floored CTC path, as the lead; a window never starts
    before the previous block's end, so background lines keep their order. Returns the blocks
    with their token indexes (input order), line sizes and spans (global frames)."""
    background = aligner_model.last_background_log_probs
    frames = background.shape[0]
    margin = round(margin_seconds / frame_shift)
    vocabulary = aligner_model.tokenizer.get_vocab()
    vowels = {vocabulary[letter] for letter in "aeiouy" if letter in vocabulary}
    letter_cap = max(1, round(min_letter_ms / 1000 / frame_shift)) if min_letter_ms > 0 else 1
    token_cap = round(min_token_ms / 1000 / frame_shift) if min_token_ms > 0 else 0
    lead_line_frames = [
        (lead_spans[start][0].start, lead_spans[end - 1][-1].end) for start, end in lead_ranges
    ]
    blocks, lead_seen = [], 0
    for line_index, flag, tokens in plan:
        if not flag:
            lead_seen += 1
            continue
        if (
            blocks
            and blocks[-1]["lead_before"] == lead_seen
            and blocks[-1]["last_line"] == line_index - 1
        ):
            blocks[-1]["tokens"].extend(tokens)
            blocks[-1]["lines"].append(len(tokens))
        else:
            blocks.append(
                {"lead_before": lead_seen, "tokens": list(tokens), "lines": [len(tokens)]}
            )
        blocks[-1]["last_line"] = line_index
    background_end = 0  # background lines stay in order among themselves
    for block in blocks:
        before, after = block["lead_before"] - 1, block["lead_before"]
        start = max(0, lead_line_frames[before][0] - margin) if before >= 0 else 0
        stop = (
            min(frames, lead_line_frames[after][1] + margin)
            if after < len(lead_line_frames)
            else frames
        )
        groups = [all_encoded[i] for i in block["tokens"]]
        start = max(start, background_end)
        stop = max(stop, min(frames, start + 2 * sum(map(len, groups))))
        window = background[start:stop].to(torch.float32)
        try:
            spans = align_with_minimum_durations(
                window,
                groups,
                blank=aligner_model.blank,
                minimum_frames=token_minimum_frames(groups, letter_cap, token_cap, vowels),
                blank_free_tokens=blank_free_tokens,
            )
        except RuntimeError:
            spans = align_with_minimum_durations(
                window,
                groups,
                blank=aligner_model.blank,
                minimum_frames=[1] * sum(map(len, groups)),
                blank_free_tokens=blank_free_tokens,
            )
        block["spans"] = [
            [TokenSpan(s.token, s.start + start, s.end + start, s.score) for s in g] for g in spans
        ]
        background_end = block["spans"][-1][-1].end
    return blocks


def _place_skipped_groups(
    spans,
    encoded,
    line_ranges,
    emission,
    blank,
    frame_shift,
    min_letter_ms,
    min_token_ms,
    vowels,
    blank_free_tokens,
    margin_seconds=1.0,
):
    """Optional brackets: decode each run of groups the lead path left out on its own.

    A run's window starts where the lead word before it ends (an echo answers what was just sung)
    and ends with the next lead line (plus ``margin_seconds``), so it may overlap that line; runs
    stay in order among themselves. ``emission`` is the background head when the checkpoint has
    one, else the lead's. Returns the filled spans and the indexes of the placed groups."""
    frames = emission.shape[0]
    margin = round(margin_seconds / frame_shift)
    letter_cap = max(1, round(min_letter_ms / 1000 / frame_shift)) if min_letter_ms > 0 else 1
    token_cap = round(min_token_ms / 1000 / frame_shift) if min_token_ms > 0 else 0
    line_of = {g: k for k, (start, end) in enumerate(line_ranges) for g in range(start, end)}
    spans, placed, previous_end = list(spans), [], 0
    g = 0
    while g < len(spans):
        if spans[g]:
            g += 1
            continue
        last = g
        while last + 1 < len(spans) and not spans[last + 1]:
            last += 1
        before = next((spans[k][-1].end for k in range(g - 1, -1, -1) if spans[k]), 0)
        line = line_of.get(last, len(line_ranges) - 1)
        later = [
            spans[k][-1].end
            for k in range(last + 1, line_ranges[min(line + 1, len(line_ranges) - 1)][1])
            if spans[k]
        ]
        start = max(before, previous_end)
        stop = min(frames, (max(later) if later else frames) + margin)
        groups = encoded[g : last + 1]
        stop = max(stop, min(frames, start + 2 * sum(map(len, groups))))
        start = min(start, max(0, stop - 2 * sum(map(len, groups))))
        window = emission[start:stop].to(torch.float32)
        try:
            found = align_with_minimum_durations(
                window,
                groups,
                blank=blank,
                minimum_frames=token_minimum_frames(groups, letter_cap, token_cap, vowels),
                blank_free_tokens=blank_free_tokens,
            )
        except RuntimeError:
            found = align_with_minimum_durations(
                window,
                groups,
                blank=blank,
                minimum_frames=[1] * sum(map(len, groups)),
                blank_free_tokens=blank_free_tokens,
            )
        for k, group in zip(range(g, last + 1), found):
            spans[k] = [TokenSpan(s.token, s.start + start, s.end + start, s.score) for s in group]
            placed.append(k)
        previous_end = spans[last][-1].end
        g = last + 1
    return spans, placed


def _merge_background_results(
    lead_results,
    blocks,
    all_tokens,
    frame_shift,
    speed,
    timing_offset_ms,
    frame_times,
    onset_offsets_ms,
):
    """Display results of the background blocks, merged into the lead's in input token order."""
    background_results = {}
    for block in blocks:
        ranges, offset = [], 0
        for count in block["lines"]:
            ranges.append((offset, offset + count))
            offset += count
        tokens = [all_tokens[i] for i in block["tokens"]]
        token_offsets = (
            [float(onset_offsets_ms.get(onset_class(token), 0.0)) / 1000.0 for token in tokens]
            if onset_offsets_ms
            else None
        )
        results = _display_span_results(
            tokens,
            block["spans"],
            ranges,
            frame_shift * speed,
            timing_offset_ms / 1000.0,
            frame_times,
            None,
            token_offsets,
        )
        for index, result in zip(block["tokens"], results):
            result["voice"] = "background"
            background_results[index] = result
    lead_iter = iter(lead_results)
    return [
        background_results[index] if index in background_results else next(lead_iter)
        for index in range(len(all_tokens))
    ]


def _wav2vec_aligner(
    model: str, use_gpu: bool, fusion: str = "off", fusion_model: str | None = None
) -> Wav2Vec2ForcedAligner:
    key = (model, bool(use_gpu), fusion, fusion_model)
    cached = _WAV2VEC_ALIGNER_CACHE.get(key)
    if cached is None:
        cached = Wav2Vec2ForcedAligner(
            model, use_gpu=use_gpu, fusion=fusion, fusion_model=fusion_model
        )
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
    exact_frame_clock=False,
    subframe_refinement=False,
    onset_offsets_ms=None,
    edge_weight=0.0,
    vocal_activity_source="energy",
    fusion="off",
    fusion_model=None,
    vocal_activity_file=None,
    blank_free_tokens=False,
    background_lines=None,
    optional_tokens=None,
):
    """Align the complete song from the audio alone, in one global CTC pass.

    ``token_lines`` gives the lyric lines (for line ranges and pacing); no source times are used.

    Opt-in experiments (all off by default, which reproduces the shipped decoder exactly):
    ``exact_frame_clock`` (E3) reports each frame at its true receptive-field centre instead of
    duration / frames; ``subframe_refinement`` (S2) moves each token start by the interpolated
    zero crossing of its first letter's log-odds; ``onset_offsets_ms`` (S2) maps onset classes
    (``ctc_segmentation.ONSET_CLASSES``) to an extra offset added to ``timing_offset_ms``;
    ``edge_weight`` (S1) uses the checkpoint's edge head in the duration-floor Viterbi;
    ``vocal_activity_source`` picks what ``vocal_activity_weight`` penalises against: ``energy``
    (loudness of the input, scaled per song) or ``vad`` (the checkpoint's auxiliary vocal-activity
    head, a per-frame singing probability; needs ``aux_heads.safetensors``);
    ``fusion`` (§5f) loads a MuQ fusion checkpoint's frozen MuQ branch (``"on"``), runs it
    as the plain encoder (``"gate0"``, ablation) or refuses it (``"off"``, the default, which
    changes nothing for every other checkpoint); ``fusion_model`` overrides the MuQ id/path.
    ``background_lines`` (B0, opt-in) flags token lines sung by a background voice: they leave the
    lead path and are decoded on the checkpoint's background head, block by block, inside the span
    of their neighbouring lead lines, so they may overlap the lead. Results keep the input order.
    ``optional_tokens`` (opt-in, one flag per acoustic token) marks inline backing vocals: the lead
    path may leave them out (``align_with_minimum_durations(optional_groups=...)``); the ones it
    leaves out are placed on their own, may overlap the lead, and come back with ``"stream": 2``.
    """
    if vocal_activity_source not in VOCAL_ACTIVITY_SOURCES:
        raise ValueError(f"vocal_activity_source must be one of {VOCAL_ACTIVITY_SOURCES}")
    if fusion not in ("off", "on", "gate0"):
        raise ValueError("fusion must be off, on or gate0")
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
    aligner_model = _wav2vec_aligner(hf_model_id, use_gpu, fusion, fusion_model)
    encoded = aligner_model.tokenize(acoustic_tokens)
    if len(encoded) != len(acoustic_tokens) or any(not group for group in encoded):
        raise RuntimeError("tokenizer omitted one or more acoustic tokens")

    emission, _, frame_shift = aligner_model.infer(waveform, sample_rate)
    original_inference_passes = aligner_model.last_inference_passes
    all_token_lines, background_plan = token_lines, None
    if background_lines and any(background_lines):
        if len(background_lines) != len(token_lines):
            raise ValueError("background_lines needs one flag per token line")
        if aligner_model.last_background_log_probs is None:
            raise RuntimeError("background_lines needs a checkpoint with a background head")
        if all(background_lines):
            raise ValueError("background_lines leaves no lead line to align")
        # Token indexes (in input order) of the lead and background lines.
        line_tokens, offset = [], 0
        for line in token_lines:
            count = sum(1 for token in line if token)
            line_tokens.append(list(range(offset, offset + count)))
            offset += count
        background_plan = [
            (index, flag, line_tokens[index]) for index, flag in enumerate(background_lines)
        ]
        lead_indexes = [i for _, flag, tokens in background_plan if not flag for i in tokens]
        all_tokens, all_encoded = acoustic_tokens, encoded
        token_lines = [line for line, flag in zip(token_lines, background_lines) if not flag]
        acoustic_tokens = [all_tokens[i] for i in lead_indexes]
        encoded = [all_encoded[i] for i in lead_indexes]
    all_optional = optional_tokens  # in input token order, for the text head
    if background_plan is not None and optional_tokens is not None:
        optional_tokens = [optional_tokens[i] for i in lead_indexes]
    if optional_tokens is not None and len(optional_tokens) != len(acoustic_tokens):
        raise ValueError("optional_tokens needs one flag per acoustic token")
    optional_groups = {i for i, flag in enumerate(optional_tokens or []) if flag} or None
    line_acoustic_ranges = []
    acoustic_offset = 0
    for line in token_lines:
        line_token_count = sum(1 for token in line if token)
        if line_token_count:
            line_acoustic_ranges.append((acoustic_offset, acoustic_offset + line_token_count))
            acoustic_offset += line_token_count

    def _decode_lead(emission):
        """Lead path (activity penalty, pace, duration floors) on ``emission``."""
        scoring_emission = emission
        if vocal_activity_weight > 0:
            # Decode on letters made costlier where the stem is quiet (token scores keep the
            # model's own probabilities).
            if vocal_activity_source == "vad":
                activity = aligner_model.vad_activity()
                if activity.shape[0] != emission.shape[0]:
                    raise RuntimeError("VAD head frames do not match the emission lattice")
            elif vocal_activity_source == "external":
                # Per-20 ms singing activity computed by another model on the same audio (e.g. the
                # w2v-BERT VAD head on the full mix, for checkpoints without their own head): resampled
                # by time onto this lattice, then scaled per song like the other sources.
                if not vocal_activity_file:
                    raise ValueError("vocal_activity_source='external' needs vocal_activity_file")
                import numpy as np

                values = torch.from_numpy(np.load(vocal_activity_file).astype("float32"))
                activity = torch.nn.functional.interpolate(
                    values[None, None], size=emission.shape[0], mode="linear", align_corners=False
                )[0, 0]
                low, high = torch.quantile(activity, 0.1), torch.quantile(activity, 0.9)
                activity = ((activity - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
            else:
                activity = vocal_activity(
                    resample(waveform, sample_rate, 16000).mean(0), emission.shape[0]
                )
            emission = with_vocal_activity(
                emission, activity, aligner_model.blank, vocal_activity_weight
            )
        duration_floor = None
        gap_cost = float(os.environ.get("FA_KARA_IN_LINE_GAP_COST", "0"))
        has_floors = min_letter_ms > 0 or min_token_ms > 0
        use_extended = (
            has_floors
            or blank_free_tokens
            or text_share is not None
            or edge_weight > 0
            or gap_cost > 0
        )
        if use_extended:
            vocabulary = aligner_model.tokenizer.get_vocab()
            vowels = {vocabulary[letter] for letter in "aeiouy" if letter in vocabulary}
            letter_cap = max(1, round(min_letter_ms / 1000 / frame_shift))
            token_cap = round(min_token_ms / 1000 / frame_shift)
            if has_floors and duration_pace_ratio > 0:
                initial_spans = align_with_minimum_durations(
                    emission,
                    encoded,
                    blank=aligner_model.blank,
                    minimum_frames=[1 for group in encoded for _ in group],
                    blank_free_tokens=blank_free_tokens,
                    optional_groups=optional_groups,
                )
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
            edge_logits = aligner_model.last_edge_logits if edge_weight > 0 else None
            if edge_weight > 0 and edge_logits is None:
                raise RuntimeError("edge_weight needs a checkpoint with an edge head")
            decode_emission, decode_groups, letter_of = emission, encoded, None
            if text_share is not None:
                # §5j bonus: one pseudo-label per (token, letter) = the letter's column + beta x the
                # head's log P(token | singing) for that token; mapped back to letters below.
                letter_of = [letter for group in encoded for letter in group]
                # The head's lead columns are its lead tokens; optional brackets are backing words to
                # it, so on this path they are scored by its backing voice (the columns appended last).
                # A flat zero there would make them the cheapest place to wait.
                owner, column, optional_column = [], 0, text_share.shape[1]
                for k, group in enumerate(encoded):
                    if optional_groups and k in optional_groups:
                        owner.extend([optional_column] * len(group))
                        optional_column += 1
                    else:
                        owner.extend([column] * len(group))
                        column += 1
                share = (
                    text_share
                    if optional_share is None
                    else torch.cat((text_share, optional_share), 1)
                )
                decode_emission = torch.cat(
                    (
                        emission[:, letter_of] + text_bonus * share[:, owner].to(emission),
                        emission[:, aligner_model.blank : aligner_model.blank + 1],
                    ),
                    1,
                )
                decode_groups, cursor = [], 0
                for group in encoded:
                    decode_groups.append(list(range(cursor, cursor + len(group))))
                    cursor += len(group)
            decode_options = dict(
                blank=len(letter_of) if letter_of is not None else aligner_model.blank,
                acoustic_token_groups=encoded,
                edge_logits=edge_logits,
                edge_weight=edge_weight,
                blank_free_tokens=blank_free_tokens,
                # Opt-in experiment: cost per blank frame inside a line (gaps go between lines).
                line_ranges=line_acoustic_ranges,
                in_line_gap_cost=gap_cost,
                optional_groups=optional_groups,
            )
            try:
                initial_spans = align_with_minimum_durations(
                    decode_emission, decode_groups, minimum_frames=minimums, **decode_options
                )
            except RuntimeError:
                if not has_floors:
                    raise
                initial_spans = align_with_minimum_durations(
                    decode_emission,
                    decode_groups,
                    minimum_frames=[1 for group in encoded for _ in group],
                    **decode_options,
                )
                duration_floor = "too_dense_one_frame_fallback"
            if letter_of is not None:
                initial_spans = [
                    [TokenSpan(letter_of[s.token], s.start, s.end, s.score) for s in group]
                    for group in initial_spans
                ]
            duration_floor = duration_floor or (
                {
                    "letter_ms": min_letter_ms,
                    "token_ms": min_token_ms,
                    "pace_ratio": duration_pace_ratio,
                }
                if has_floors
                else None
            )
        else:
            initial_spans = align_ctc(emission, encoded, blank=aligner_model.blank)
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
        skipped = []
        if optional_groups and any(not group for group in initial_spans):
            vocabulary = aligner_model.tokenizer.get_vocab()
            background = aligner_model.last_background_log_probs
            initial_spans, skipped = _place_skipped_groups(
                initial_spans,
                encoded,
                line_acoustic_ranges,
                background if background is not None else raw_emission,
                aligner_model.blank,
                frame_shift,
                min_letter_ms,
                min_token_ms,
                {vocabulary[letter] for letter in "aeiouy" if letter in vocabulary},
                blank_free_tokens,
            )
        return emission, scoring_emission, initial_spans, duration_floor, skipped

    raw_emission = emission
    text_share, optional_share, text_bonus = (
        None,
        None,
        float(os.environ.get("FA_KARA_TEXT_BONUS", "0")),
    )
    if text_bonus > 0 and aligner_model.text_aligner is not None:
        # The whole song's tokens in text order (backing lines tagged by background_lines, as B0).
        source_lines = all_token_lines
        flags = background_lines if background_plan is not None else [False] * len(source_lines)
        groups_all = all_encoded if background_plan is not None else encoded
        voices, lines, cursor = [], [], 0
        for line_index, (line, flag) in enumerate(zip(source_lines, flags)):
            for _ in [t for t in line if t]:
                # Inline brackets are backing words to the head, as in training.
                voices.append(
                    1 if flag or (all_optional is not None and all_optional[cursor]) else 0
                )
                lines.append(line_index)
                cursor += 1
        inline = [j for j, flag in enumerate(all_optional or []) if flag]
        text_share = aligner_model.text_token_share(
            raw_emission, groups_all, voices, lines, backing=inline or None
        )
        if inline:
            text_share, optional_share = text_share
    emission, scoring_emission, initial_spans, duration_floor, skipped = _decode_lead(emission)
    background_blocks = None
    if background_plan is not None:
        # B0: place the background lines on their head, then decode the lead again with
        # their frames free to be silence for the lead (the main head hears echoes too, so
        # the lead would otherwise hold its last word through them).
        background_blocks = _decode_background_blocks(
            background_plan,
            all_encoded,
            initial_spans,
            line_acoustic_ranges,
            aligner_model,
            frame_shift,
            min_letter_ms,
            min_token_ms,
            blank_free_tokens,
        )
        explained = raw_emission.clone()
        for block in background_blocks:
            for group in block["spans"]:
                explained[group[0].start : group[-1].end, aligner_model.blank] = 0.0
        emission, scoring_emission, initial_spans, duration_floor, skipped = _decode_lead(explained)

    frame_times = None
    if exact_frame_clock:
        if aligner_model.last_frame_times is None:
            raise RuntimeError("exact frame clock unavailable for this lattice")
        frame_times = aligner_model.last_frame_times * speed
    fractions = None
    if subframe_refinement:
        fractions = refine_group_starts(scoring_emission, initial_spans, blank=aligner_model.blank)
    token_offsets = None
    if onset_offsets_ms:
        token_offsets = [
            float(onset_offsets_ms.get(onset_class(token), 0.0)) / 1000.0
            for token in acoustic_tokens
        ]
    diagnostics = {
        "source_timeline": "ignored",
        "minimum_duration": duration_floor,
        "in_line_gap_cost": float(os.environ.get("FA_KARA_IN_LINE_GAP_COST", "0")),
        "text_bonus_requested": text_bonus,
        "text_bonus_status": (
            "applied"
            if text_share is not None
            else "unavailable_no_text_head"
            if text_bonus > 0
            else "disabled"
        ),
        "timing_offset_ms": timing_offset_ms,
        "vocal_activity_weight": vocal_activity_weight,
        "ctc_blank_hold_assignment": "bounded_display_extension_not_silence_aware",
        "inference_audio_speed": speed,
    }
    if blank_free_tokens:
        diagnostics["blank_free_tokens"] = True
    if text_share is not None:
        diagnostics["text_bonus"] = text_bonus
    if exact_frame_clock or subframe_refinement or onset_offsets_ms or edge_weight > 0:
        diagnostics.update(
            exact_frame_clock=bool(exact_frame_clock),
            subframe_refinement=bool(subframe_refinement),
            onset_offsets_ms=dict(onset_offsets_ms) if onset_offsets_ms else None,
            edge_weight=edge_weight,
        )
    if vocal_activity_source != "energy" or aligner_model.backbone != "wav2vec2":
        diagnostics.update(
            vocal_activity_source=vocal_activity_source, backbone=aligner_model.backbone
        )
    # Frame k is reported at k * shift. Measured against reference word starts, those times
    # run early (frame centres, plus CTC firing ahead of the sung onset): calibrated offset.
    results = _display_span_results(
        acoustic_tokens,
        initial_spans,
        line_acoustic_ranges,
        frame_shift * speed,
        timing_offset_ms / 1000.0,
        frame_times,
        fractions,
        token_offsets,
        frozenset(skipped),
    )
    for index in skipped:
        results[index]["stream"] = 2
    if optional_groups:
        diagnostics["optional_brackets"] = {
            "optional_tokens": len(optional_groups),
            "second_stream": len(skipped),
            "head": "background_head"
            if aligner_model.last_background_log_probs is not None
            else "lead",
        }
    if background_plan is not None:
        results = _merge_background_results(
            results,
            background_blocks,
            all_tokens,
            frame_shift,
            speed,
            timing_offset_ms,
            frame_times,
            onset_offsets_ms,
        )
        diagnostics["background_decoding"] = {
            "head": "background_head",
            "lines": sum(1 for _, flag, _ in background_plan if flag),
        }
        token_lines = all_token_lines
    for result in results:
        result["inference_passes"] = original_inference_passes
        result["source_diagnostics"] = diagnostics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
