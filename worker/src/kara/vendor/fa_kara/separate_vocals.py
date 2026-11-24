"""Persistent Demucs vocal separation for the FA-Kara worker."""

import os
import random

import torch
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio

_DEMUCS_MODEL = None
_DEMUCS_MODEL_NAME = None
_SUPPORTED_MODELS = {"htdemucs", "htdemucs_ft"}


def demucs_model_name():
    model_name = os.environ.get("FA_KARA_DEMUCS_MODEL", "htdemucs_ft").strip()
    if model_name not in _SUPPORTED_MODELS:
        supported = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ValueError(f"FA_KARA_DEMUCS_MODEL must be one of: {supported}")
    return model_name


def _prepare_audio(audio_channels, sample_rate, use_gpu):
    """Return a mono vocal stem while retaining the selected model for later jobs."""
    global _DEMUCS_MODEL, _DEMUCS_MODEL_NAME
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model_name = demucs_model_name()
    if _DEMUCS_MODEL is None or _DEMUCS_MODEL_NAME != model_name:
        _DEMUCS_MODEL = pretrained.get_model(model_name)
        _DEMUCS_MODEL.eval()
        _DEMUCS_MODEL.to(device)
        _DEMUCS_MODEL_NAME = model_name
    model = _DEMUCS_MODEL
    waveform = torch.as_tensor(audio_channels.T, dtype=torch.float32)
    waveform = convert_audio(waveform, sample_rate, model.samplerate, model.audio_channels)
    reference = waveform.mean(0)
    reference_mean = reference.mean()
    reference_std = reference.std().clamp_min(1e-8)
    normalized = (waveform - reference_mean) / reference_std
    return model, device, normalized, reference_mean, reference_std


def separate_vocals(audio_channels, sample_rate, *, use_gpu=True):
    """Return a mono vocal stem while retaining the selected model for later jobs."""
    model, device, normalized, reference_mean, reference_std = _prepare_audio(
        audio_channels, sample_rate, use_gpu,
    )
    with torch.inference_mode():
        sources = apply_model(
            model, normalized.unsqueeze(0), device=device,
            shifts=1, split=True, overlap=0.25, progress=False,
        )[0]
    vocals = sources[model.sources.index("vocals")] * reference_std + reference_mean
    return vocals.mean(0).cpu().numpy(), int(model.samplerate)


def separate_vocals_single_checkpoint(audio_channels, sample_rate, *, use_gpu=True):
    """Produce the exact bagged vocal source without computing zero-weight checkpoints."""
    model, device, normalized, reference_mean, reference_std = _prepare_audio(
        audio_channels, sample_rate, use_gpu,
    )
    vocal_index = model.sources.index("vocals")
    if _DEMUCS_MODEL_NAME != "htdemucs_ft":
        with torch.inference_mode():
            sources = apply_model(
                model, normalized.unsqueeze(0), device=device,
                shifts=1, split=True, overlap=0.25, progress=False,
            )[0]
        vocals = sources[vocal_index] * reference_std + reference_mean
        return vocals.mean(0).cpu().numpy(), int(model.samplerate)
    weighted_models = [
        (index, submodel, weights[vocal_index])
        for index, (submodel, weights) in enumerate(zip(model.models, model.weights))
        if weights[vocal_index]
    ]
    if len(weighted_models) != 1 or weighted_models[0][2] != 1:
        raise RuntimeError("The selected Demucs model does not have one vocal checkpoint")
    model_index, vocal_model, _ = weighted_models[0]
    max_shift = int(0.5 * vocal_model.samplerate)
    for skipped_model in model.models[:model_index]:
        offset = random.randint(0, max_shift)
        shifted_length = normalized.shape[-1] + max_shift - offset
        segment_length = int(skipped_model.samplerate * skipped_model.segment)
        stride = int(0.75 * segment_length)
        segment_count = (shifted_length + stride - 1) // stride
        transformer = skipped_model.crosstransformer
        for _ in range(segment_count):
            random.randrange(transformer.sin_random_shift + 1)
    with torch.inference_mode():
        sources = apply_model(
            vocal_model, normalized.unsqueeze(0), device=device,
            shifts=1, split=True, overlap=0.25, progress=False,
        )[0]
    vocals = sources[vocal_index] * reference_std + reference_mean
    return vocals.mean(0).cpu().numpy(), int(model.samplerate)
