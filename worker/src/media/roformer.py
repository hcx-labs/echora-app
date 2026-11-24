"""Local-only Mel-Band-Roformer vocal separation at 44.1 kHz stereo.

No resampling, normalization, clipping or network fallback. See vendor/roformer/NOTICE
for source attribution, model licensing and author permission references. Full-track buffers stay
on CPU; only one inference chunk and the model reside on the accelerator.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import gc

import numpy as np
import torch

from core.revisions import (
    SEPARATION_REVISION,
    SEPARATOR_CHUNK_SIZE as _CHUNK_SIZE,
    SEPARATOR_MODEL_FILENAME as MODEL_FILENAME,
    SEPARATOR_MODEL_ID as MODEL_ID,
    SEPARATOR_MODEL_REVISION as MODEL_REVISION,
    SEPARATOR_OVERLAP as _OVERLAP,
)

SAMPLE_RATE = 44100
# Exact upstream model config; no YAML loader or training config dependency.
_MODEL_CONFIG = dict(
    dim=384,
    depth=6,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_bands=60,
    dim_head=64,
    heads=8,
    attn_dropout=0,
    ff_dropout=0,
    flash_attn=True,
    dim_freqs_in=1025,
    sample_rate=SAMPLE_RATE,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147,
    multi_stft_normalized=False,
)


def _new_model():
    from .vendor.roformer.mel_band_roformer import MelBandRoformer

    return MelBandRoformer(**_MODEL_CONFIG)


def _load_weights(model):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        filename=MODEL_FILENAME,
        local_files_only=True,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    finally:
        del state


def _predict(model, part: np.ndarray, device: torch.device) -> np.ndarray:
    """Drop device tensor references even when a retained traceback escapes."""
    batch = prediction = None
    try:
        batch = torch.from_numpy(np.ascontiguousarray(part)).unsqueeze(0).to(device)
        amp = (
            torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        )
        with torch.inference_mode(), amp:
            prediction = model(batch)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != batch.shape:
            raise ValueError("Roformer returned an invalid chunk shape")
        result = prediction.detach().to(device="cpu", dtype=torch.float32).numpy().copy()[0]
        if not np.isfinite(result).all():
            raise ValueError("Roformer returned non-finite audio")
        return result
    finally:
        batch = prediction = None


def _demix(model, waveform: np.ndarray, device: torch.device, check: Callable[[], None]):
    chunk = _CHUNK_SIZE
    step = chunk // _OVERLAP
    fade = chunk // 10
    border = chunk - step
    original_length = len(waveform)
    # Pad short clips before the upstream reflect/crop decision. Track the decision
    # explicitly: upstream erroneously decides cropping from the padded length.
    mix = np.pad(waveform.T, ((0, 0), (0, max(0, chunk - original_length))))
    reflected = mix.shape[1] > 2 * border
    if reflected:
        mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")
    total = mix.shape[1]
    # float64 accumulation avoids overflow for finite float32 model outputs.
    result = np.zeros((2, total), dtype=np.float64)
    counter = np.zeros(total, dtype=np.float64)
    window = np.ones(chunk, dtype=np.float32)
    window[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
    window[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
    for start in range(0, total, step):
        check()
        part = mix[:, start : start + chunk]
        length = part.shape[1]
        if length < chunk:
            mode = "reflect" if length > chunk // 2 + 1 else "constant"
            part = np.pad(part, ((0, 0), (0, chunk - length)), mode=mode)
        prediction = _predict(model, part, device)
        weights = window.copy()
        if start == 0:
            weights[:fade] = 1
        if start + chunk >= total:
            weights[-fade:] = 1
        result[:, start : start + length] += (
            prediction[:, :length].astype(np.float64) * weights[:length]
        )
        counter[start : start + length] += weights[:length]
    check()
    if not np.all(counter > 0):
        raise ValueError("Roformer overlap left uncovered samples")
    result /= counter
    offset = border if reflected else 0
    return np.ascontiguousarray(result[:, offset : offset + original_length].T, dtype=np.float32)


class Separator:
    """One loaded Roformer: load once, separate any number of tracks, then close."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _new_model()
        try:
            _load_weights(model)
            model.eval()
            model.to(self.device)
        except BaseException:
            model.cpu()
            raise
        self.model = model

    def separate(
        self, waveform: np.ndarray, check: Callable[[], None] = lambda: None
    ) -> np.ndarray:
        result = _demix(self.model, waveform, self.device, check)
        if result.shape != waveform.shape or result.dtype != np.float32:
            raise ValueError("Roformer returned an invalid waveform shape or dtype")
        if not np.isfinite(result).all():
            raise ValueError("Roformer returned non-finite audio")
        return result

    def close(self) -> None:
        try:
            # Move parameters to CPU even if an exception traceback retains the model.
            if self.model is not None:
                self.model.cpu()
        finally:
            self.model = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


# The Separator a batch phase holds; separate_vocals reuses it instead of loading per track.
_phase: ContextVar[dict | None] = ContextVar("roformer_phase", default=None)


@contextmanager
def separation_phase() -> Iterator[None]:
    """Keep one Roformer loaded for a batch phase: loaded on first use, released at the end.

    Nothing loads when no track in the phase needs separating (all cached, or none selected).
    """
    holder: dict = {"separator": None}
    token = _phase.set(holder)
    try:
        yield
    finally:
        _phase.reset(token)
        if holder["separator"] is not None:
            holder["separator"].close()


@contextmanager
def using(separator: Separator) -> Iterator[None]:
    """Separate with an already loaded Separator (one a Modal container keeps between calls)."""
    token = _phase.set({"separator": separator})
    try:
        yield
    finally:
        _phase.reset(token)


def separate_vocals(
    waveform: np.ndarray,
    check: Callable[[], None] = lambda: None,
) -> np.ndarray:
    """Separate finite float32 (frames, 2) audio; cancellation exceptions propagate.

    Empty audio returns a fresh empty array without loading weights. Inside a
    separation phase, the phase's model is reused; otherwise each call loads a fresh
    model from the pinned local HF cache and releases it in finally. The caller must
    already have cached the checkpoint; missing cache is an error.
    """
    if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32:
        raise ValueError("waveform must be a float32 numpy array")
    if waveform.ndim != 2 or waveform.shape[1] != 2:
        raise ValueError("waveform must have shape (frames, 2)")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform must contain only finite samples")
    check()
    if not len(waveform):
        return waveform.copy()
    holder = _phase.get()
    if holder is not None:
        if holder["separator"] is None:
            holder["separator"] = Separator()
        check()
        return holder["separator"].separate(waveform, check)
    separator = Separator()
    try:
        check()
        return separator.separate(waveform, check)
    finally:
        separator.close()
