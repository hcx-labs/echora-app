"""MuQ fusion branch for the FA-Kara aligner (V1_LARGE_PLAN.md §5f).

A frozen MuQ music encoder (OpenMuQ/MuQ-large-msd-iter: 12 conformer blocks x 1024, 128-bin mel
at 24 kHz, hop 240, conv subsampling x4 -> one feature per 40 ms) hears the full mix next to the
omniASR/MMS encoder. One intermediate MuQ layer is projected (Linear 1024 -> hidden) and added,
through a gate initialised at zero, to the encoder's last hidden state right before ``lm_head``
(and the S1 edge head). With the gate at zero the fused model is bit-identical to the plain one.

Shared by the trainer (``train_forced_aligner.py``), the validation harness
(``eval_validation_s3.py``) and production inference (``align_yohane.Wav2Vec2ForcedAligner``),
so the three compute the same features from the same 16 kHz waveform: the branch resamples the
model's own 16 kHz input to 24 kHz itself (no second decode of the source), so what MuQ hears at
inference is exactly what it heard in training.

Frame clock. torchaudio's MelSpectrogram is centred (``center=True``), so mel frame m sits at
sample 240 m; MuQ drops the last mel frame and subsamples twice with 3x3 stride-2 padding-1
convolutions, whose output j is centred on input 2 j. MuQ frame j is therefore centred at
960 j samples = 40 j ms (offset 0). The branch is sampled onto the aligner's 20 ms clock by linear
interpolation at each aligner frame's receptive-field centre (12.47 ms + 20 k ms for wav2vec2,
17.47 ms + 20 k ms for w2v-BERT), clamped at the ends, so lengths always match the logits.

LoRA (opt-in, 2026-10-18): ``fusion_config(..., lora=lora_config(...))`` adds low-rank adapters
(``MuQLoRA``, saved as ``fusion_lora.*``) to the attention q/v (optionally also the FFN) projections of
the top N conformer blocks up to the fused layer. The MuQ base weights stay frozen; B starts at zero,
so a fresh LoRA leaves the model exactly as it was. A LoRA-adapted MuQ is a derivative of the
CC BY-NC 4.0 weights and stays non-commercial.

Licences: the ``muq`` package (code) is MIT; the MuQ-large-msd-iter weights are CC BY-NC 4.0
(non-commercial). A fusion checkpoint depends on those weights at inference.
"""

from __future__ import annotations

import math
import os

import torch

MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
MUQ_SAMPLE_RATE = 24_000
MUQ_HOP = 240
MUQ_FOLD = 4  # label_rate 25 -> two stride-2 convolutions
MUQ_STRIDE_SECONDS = MUQ_HOP * MUQ_FOLD / MUQ_SAMPLE_RATE  # 0.04
MUQ_OFFSET_SECONDS = 0.0  # frame j centre at 40 j ms (see the module docstring)
MUQ_DIM = 1024
MUQ_BLOCKS = 12
# ``layer`` of a fusion config: 0 = conformer input (after the conv front end), 1..12 = output of
# conformer block k, 13 = block 12 after the encoder's final layer norm (MuQ's published output).
MUQ_LAYERS = tuple(range(MUQ_BLOCKS + 2))
MUQ_FINAL_LAYER = MUQ_BLOCKS + 1
# Trainer default: the layer whose linear probe best separates sung from blank frames on full-mix
# songs of the 100k validation split (data/finetune-curation/probe_muq_layers.py, 2026-10-18, three
# probes of 20 songs: block 11 AUC 0.976 / 0.977 / 0.921 (best or within 0.001 of the best each
# time); blocks 10-13 form the top band; block 6 0.977 / 0.976 / 0.915; block 0 0.90 / 0.91 / 0.85).
DEFAULT_LAYER = 11


class _Captured(Exception):
    """Raised by the deepest wanted hook so the conformer stops there (blocks past the fused
    layer are never run)."""


FUSION_KINDS = ("muq",)
FUSION_MODES = ("off", "on", "gate0")
FUSION_PREFIX = "fusion."
LORA_PREFIX = "fusion_lora."


def muq_frame_count(samples_24k: int) -> int:
    """MuQ frames for ``samples_24k`` input samples: floor(N / 240) mel frames (centred STFT
    minus the dropped last frame), then ceil(n / 2) twice."""
    mel = samples_24k // MUQ_HOP
    return (((mel + 1) // 2) + 1) // 2


def fusion_config(
    layer: int, gate: str = "scalar", model: str = MUQ_MODEL_ID, lora: dict | None = None
) -> dict:
    """The ``config.fusion`` record a fusion checkpoint carries (``lora``: a ``lora_config`` record)."""
    if layer not in MUQ_LAYERS:
        raise ValueError(f"fusion layer must be one of {MUQ_LAYERS}")
    if gate not in ("scalar", "channel"):
        raise ValueError("fusion gate must be 'scalar' or 'channel'")
    record = {
        "kind": "muq",
        "model": model,
        "layer": int(layer),
        "gate": gate,
        "dim": MUQ_DIM,
        "sample_rate": MUQ_SAMPLE_RATE,
        "stride_seconds": MUQ_STRIDE_SECONDS,
        "offset_seconds": MUQ_OFFSET_SECONDS,
    }
    if lora:
        record["lora"] = lora
    return record


# ---------------------------------------------------------------------------------------------
# LoRA on the frozen MuQ branch

# Linear sub-modules of a conformer block (Wav2Vec2ConformerEncoderLayer) and their (in, out) sizes.
LORA_MODULES = {
    "self_attn.linear_q": (MUQ_DIM, MUQ_DIM),
    "self_attn.linear_v": (MUQ_DIM, MUQ_DIM),
    "ffn1.intermediate_dense": (MUQ_DIM, 4 * MUQ_DIM),
    "ffn1.output_dense": (4 * MUQ_DIM, MUQ_DIM),
    "ffn2.intermediate_dense": (MUQ_DIM, 4 * MUQ_DIM),
    "ffn2.output_dense": (4 * MUQ_DIM, MUQ_DIM),
}
LORA_TARGETS = {"qv": ("self_attn.linear_q", "self_attn.linear_v"), "qv_ffn": tuple(LORA_MODULES)}


def lora_config(
    rank: int, alpha: float, layers: int, fused_layer: int, targets: str = "qv"
) -> dict:
    """LoRA record: blocks ``fused_layer - layers + 1 .. fused_layer`` (1-based conformer blocks; the
    final-norm layer 13 counts as block 12), ``targets`` "qv" or "qv_ffn"."""
    top = min(int(fused_layer), MUQ_BLOCKS)
    if rank <= 0 or layers <= 0 or top < 1:
        raise ValueError("LoRA needs rank > 0, layers > 0 and a fused layer >= 1")
    if targets not in LORA_TARGETS:
        raise ValueError(f"LoRA targets must be one of {tuple(LORA_TARGETS)}")
    return {
        "rank": int(rank),
        "alpha": float(alpha),
        "blocks": list(range(max(1, top - int(layers) + 1), top + 1)),
        "targets": targets,
    }


class LoRALinear(torch.nn.Module):
    """``scale * B A x`` added to a frozen linear layer's output; B = 0 at start (no change)."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = torch.nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.lora_a.dtype)
        return (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale


class MuQLoRA(torch.nn.Module):
    """All LoRA adapters of one fusion checkpoint (saved as ``fusion_lora.*``). ``attach`` hooks them
    onto a ``MuQBranch``; the branch then runs with gradients through them while training."""

    def __init__(self, record: dict):
        super().__init__()
        self.record = dict(record)
        self.adapters = torch.nn.ModuleDict(
            {
                f"b{block}__{name.replace('.', '__')}": LoRALinear(
                    *LORA_MODULES[name], int(record["rank"]), float(record["alpha"])
                )
                for block in record["blocks"]
                for name in LORA_TARGETS[record["targets"]]
            }
        )

    def reset_parameters(self) -> None:
        for adapter in self.adapters.values():
            adapter.reset_parameters()

    def attach(self, branch: "MuQBranch") -> None:
        layers = branch.muq.model.conformer.layers
        for handle in getattr(branch, "lora_handles", []):
            handle.remove()
        branch.lora_handles = []
        for key, adapter in self.adapters.items():
            block, name = key.split("__", 1)
            module = layers[int(block[1:]) - 1].get_submodule(name.replace("__", "."))

            def hook(_, inputs, output, adapter=adapter):
                return output + adapter(inputs[0]).to(output.dtype)

            branch.lora_handles.append(module.register_forward_hook(hook))
        branch.lora = self


def load_fusion_lora(model_dir: str, record: dict) -> MuQLoRA:
    """The checkpoint's trained LoRA (``fusion_lora.*``) for its ``config.fusion["lora"]`` record."""
    lora = MuQLoRA(record)
    state = fusion_state_from_checkpoint(model_dir, LORA_PREFIX)
    expected = {LORA_PREFIX + k for k in lora.state_dict()}
    if set(state) != expected:
        raise RuntimeError(
            f"fusion LoRA declared in config but its weights are missing in {model_dir}"
        )
    lora.load_state_dict({k.removeprefix(LORA_PREFIX): v for k, v in state.items()})
    return lora.eval()


def to_frame_clock(
    features: torch.Tensor,
    frames: int,
    stride_seconds: float,
    offset_seconds: float,
    source_stride_seconds: float = MUQ_STRIDE_SECONDS,
    source_offset_seconds: float = MUQ_OFFSET_SECONDS,
) -> torch.Tensor:
    """Sample ``features`` (B, T, D) on ``source`` frame centres at the ``frames`` centres of the
    target clock (``offset + k * stride``) by linear interpolation, clamped at both ends."""
    count = features.shape[1]
    k = torch.arange(frames, device=features.device, dtype=torch.float64)
    position = (
        (offset_seconds + k * stride_seconds) - source_offset_seconds
    ) / source_stride_seconds
    position = position.clamp(0.0, max(count - 1, 0))
    lower = position.floor().long()
    upper = torch.clamp(lower + 1, max=max(count - 1, 0))
    weight = (position - lower.double()).to(features.dtype).view(1, frames, 1)
    return features[:, lower] * (1 - weight) + features[:, upper] * weight


class FusionAdapter(torch.nn.Module):
    """The trained part of the fusion: projection of the MuQ layer plus a zero-initialised gate
    (one scalar, or one value per hidden channel). Saved in the checkpoint as ``fusion.*``."""

    def __init__(self, hidden_size: int, source_dim: int = MUQ_DIM, gate: str = "scalar"):
        super().__init__()
        self.proj = torch.nn.Linear(source_dim, hidden_size)
        self.gate = torch.nn.Parameter(torch.zeros(hidden_size if gate == "channel" else 1))
        self.gate_kind = gate
        # Ablation (eval watcher): behave as the plain model, whatever the trained gate says.
        self.force_zero = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.gate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # MuQ runs in float32; a bf16-loaded model (eval --bf16, inference) needs them in its dtype.
        # Under training autocast this is a no-op (fp32 master weights).
        return self.gate * self.proj(features.to(self.proj.weight.dtype))


class MuQBranch:
    """The frozen MuQ encoder with hooks on its conformer blocks. Not an ``nn.Module`` on purpose:
    attached to the aligner outside its module tree so it is never trained, saved or moved by
    the Trainer (``AlignerMixin.attach_fusion_backbone``)."""

    def __init__(
        self,
        model: str = MUQ_MODEL_ID,
        device: torch.device | str = "cpu",
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ):
        from muq import MuQ

        self.model_id = model
        # A local directory as given; otherwise a hub id resolved through the local HF cache
        # (HF_HOME), which works offline once the snapshot is present.
        self.muq = MuQ.from_pretrained(model).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.muq.to(self.device)
        self.autocast_dtype = autocast_dtype
        self.captured: dict[int, torch.Tensor] = {}
        self.wanted: set[int] = set()
        self.lora: "MuQLoRA | None" = None  # MuQLoRA.attach
        conformer = self.muq.model.conformer
        # muq 0.1.0 builds the HF conformer from an EasyDict; transformers >= 5 reads the attention
        # implementation from the config at forward time, which that dict never had.
        if not hasattr(conformer.config, "_attn_implementation"):
            conformer.config._attn_implementation = "sdpa"
        if len(conformer.layers) != MUQ_BLOCKS:
            raise RuntimeError(
                f"MuQ checkpoint has {len(conformer.layers)} conformer blocks, expected {MUQ_BLOCKS}"
            )
        conformer.layers[0].register_forward_pre_hook(self._capture_pre(0))
        for index, block in enumerate(conformer.layers, 1):
            block.register_forward_hook(self._capture(index))
        conformer.layer_norm.register_forward_hook(self._capture(MUQ_FINAL_LAYER))

    def _capture_pre(self, layer: int):
        def hook(_, args):
            if layer in self.wanted:
                self.captured[layer] = args[0]
                if layer >= max(self.wanted):
                    raise _Captured

        return hook

    def _capture(self, layer: int):
        def hook(_, __, output):
            if layer in self.wanted:
                self.captured[layer] = output[0] if isinstance(output, (tuple, list)) else output
                if layer >= max(self.wanted):
                    raise _Captured

        return hook

    def to(self, device) -> "MuQBranch":
        self.device = torch.device(device)
        self.muq.to(self.device)
        return self

    def features(
        self,
        waveform_16k: torch.Tensor,
        lengths_16k: torch.Tensor | None = None,
        layers: tuple[int, ...] | list[int] = (MUQ_FINAL_LAYER,),
        sample_rate: int = 16_000,
    ) -> dict[int, torch.Tensor]:
        """MuQ hidden states (float32, B x T x 1024) of ``layers`` for a batch of raw 16 kHz
        mono waveforms (B x L, zero-padded; ``lengths_16k`` marks the valid samples). The
        aligner path always passes 16 kHz audio; ``sample_rate`` exists for the layer probe's
        native-24 kHz comparison only."""
        import torchaudio

        waveform_16k = waveform_16k.to(self.device, torch.float32)
        if waveform_16k.dim() == 1:
            waveform_16k = waveform_16k[None]
        lengths = (
            [int(n) for n in lengths_16k.tolist()]
            if lengths_16k is not None
            else [waveform_16k.shape[1]] * waveform_16k.shape[0]
        )
        self.wanted = set(int(layer) for layer in layers)
        # Gradients flow only through an attached LoRA in training mode; the base stays frozen.
        grad = self.lora is not None and self.lora.training and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad):
            return self._features(waveform_16k, lengths, sample_rate)

    def _features(self, waveform_16k, lengths, sample_rate):
        try:
            if len(set(lengths)) == 1 and lengths[0] == waveform_16k.shape[1]:
                return self._encode(waveform_16k, sample_rate)
            # Unequal lengths: each sample at its own length (an attention mask does not isolate
            # a padded sample: the conformer's depthwise convolutions read across the padding),
            # exactly as inference runs one chunk at a time; features past a sample's own last
            # frame repeat it, which is what the clock interpolation's clamping sees anyway.
            per_sample = [
                self._encode(waveform_16k[i : i + 1, :length], sample_rate)
                for i, length in enumerate(lengths)
            ]
            count = max(out[next(iter(self.wanted))].shape[1] for out in per_sample)
            return {
                layer: torch.cat(
                    [
                        torch.cat(
                            [
                                out[layer],
                                out[layer][:, -1:].expand(-1, count - out[layer].shape[1], -1),
                            ],
                            dim=1,
                        )
                        for out in per_sample
                    ]
                )
                for layer in self.wanted
            }
        finally:
            self.captured = {}
            self.wanted = set()

    def _encode(self, wave: torch.Tensor, sample_rate: int) -> dict[int, torch.Tensor]:
        """``MuQModel.get_predictions`` without its ``hidden_states`` return (transformers >= 5
        no longer hands them back from the bare conformer encoder; the hooks collect them)."""
        import torchaudio

        if sample_rate != MUQ_SAMPLE_RATE:
            wave = torchaudio.functional.resample(wave, sample_rate, MUQ_SAMPLE_RATE)
        model = self.muq.model
        self.captured = {}
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None and self.device.type == "cuda",
        ):
            x = model.preprocessing(wave, features=["melspec_2048"])
            x = model.normalize(x)
            try:
                model.conformer(model.conv(x["melspec_2048"]))
            except _Captured:
                pass
        out = {layer: self.captured[layer].float() for layer in self.wanted}
        self.captured = {}
        return out


def fusion_state_from_checkpoint(
    model_dir: str, prefix: str = FUSION_PREFIX
) -> dict[str, torch.Tensor]:
    """``fusion.*`` (or ``prefix``) tensors of a checkpoint directory (safetensors or pytorch_model.bin)."""
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    state: dict[str, torch.Tensor] = {}
    if os.path.isfile(safetensors_path):
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            state = {k: handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix)}
    else:
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            full = torch.load(bin_path, map_location="cpu")
            state = {k: v for k, v in full.items() if k.startswith(prefix)}
    return state


def load_fusion_adapter(model_dir: str, config: dict, hidden_size: int) -> FusionAdapter:
    """The checkpoint's trained adapter for its ``config.fusion`` record."""
    adapter = FusionAdapter(
        hidden_size, int(config.get("dim", MUQ_DIM)), config.get("gate", "scalar")
    )
    state = fusion_state_from_checkpoint(model_dir)
    expected = {FUSION_PREFIX + k for k in adapter.state_dict()}
    if set(state) != expected:
        raise RuntimeError(
            f"fusion declared in config but its weights are missing in {model_dir}: "
            f"have {sorted(state)}, need {sorted(expected)}"
        )
    adapter.load_state_dict({k.removeprefix(FUSION_PREFIX): v for k, v in state.items()})
    return adapter.eval()


def fusion_term(
    adapter: FusionAdapter,
    branch: MuQBranch | None,
    waveform_16k: torch.Tensor,
    lengths_16k: torch.Tensor | None,
    frames: int,
    layer: int,
    stride_seconds: float,
    offset_seconds: float,
) -> torch.Tensor | None:
    """What is added to the aligner's hidden state (B x frames x hidden), or None when the
    adapter is forced to zero (ablation) and the plain model is wanted bit for bit."""
    if adapter.force_zero:
        return None
    if branch is None:
        raise RuntimeError(
            "fusion checkpoint needs its MuQ branch (attach_fusion_backbone / fusion='on')"
        )
    source = branch.features(waveform_16k, lengths_16k, layers=(layer,))[layer]
    return adapter(to_frame_clock(source, frames, stride_seconds, offset_seconds))


def expected_feature_frames(samples_16k: int) -> int:
    """MuQ frames the branch yields for a 16 kHz input of ``samples_16k`` samples."""
    return muq_frame_count(int(math.ceil(samples_16k * MUQ_SAMPLE_RATE / 16_000)))
