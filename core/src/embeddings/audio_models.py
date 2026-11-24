from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
import time

import numpy as np
import torch


@dataclass(frozen=True)
class EmbeddingResult:
    vector: np.ndarray
    inference_ms: int
    peak_vram_bytes: int | None
    window_vectors: tuple[np.ndarray, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Model returned an invalid embedding")
    return (vector / norm).astype(np.float32)


def release_model(model: object) -> None:
    """Release a model before the next analysis phase claims RAM or VRAM."""
    loaded = getattr(model, "model", None)
    if loaded is not None and hasattr(loaded, "to"):
        loaded.to("cpu")
    if hasattr(model, "model"):
        delattr(model, "model")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class AudioEmbeddingModel(ABC):
    name: str
    revision: str
    sample_rate = 24_000
    window_seconds = 10

    @abstractmethod
    def embed_windows(self, windows: list[np.ndarray]) -> EmbeddingResult:
        """Embed ordered mono windows and return their vectors plus a track aggregate."""


class MuQMuLanModel(AudioEmbeddingModel):
    name = "muq_mulan"

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        from muq import MuQMuLan

        self.device = torch.device(device)
        self.revision = revision
        # Loading by repository ID lets huggingface_hub fall back to the
        # checkpoint's pytorch_model.bin. Its local-directory path only checks
        # for model.safetensors, which this upstream checkpoint does not ship.
        self.model = MuQMuLan.from_pretrained(model_id, revision=revision).to(self.device).eval()

    def embed_windows(self, windows: list[np.ndarray]) -> EmbeddingResult:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for window in windows:
                waveform = torch.from_numpy(window).unsqueeze(0).to(self.device)
                output = self.model(wavs=waveform)
                vectors.append(output.detach().float().cpu().numpy()[0])
        normalized = tuple(_normalize(item) for item in vectors)
        vector = _normalize(np.mean(normalized, axis=0))
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        return EmbeddingResult(vector, round((time.perf_counter() - started) * 1000), peak, normalized)


class MertModel(AudioEmbeddingModel):
    name = "mert"

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        from transformers import AutoModel

        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def embed_each(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """Return one normalized vector per window without track-level aggregation."""
        vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for window in windows:
                waveform = torch.from_numpy(window).unsqueeze(0).to(self.device)
                output = self.model(waveform, output_hidden_states=True)
                # Preserve each local window. Pooling only removes MERT's frame axis.
                pooled = output.hidden_states[-1].mean(dim=1)
                vectors.append(_normalize(pooled.detach().float().cpu().numpy()[0]))
        return vectors

    def embed_windows(self, windows: list[np.ndarray]) -> EmbeddingResult:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        vectors = self.embed_each(windows)
        vector = _normalize(np.mean(vectors, axis=0))
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        return EmbeddingResult(vector, round((time.perf_counter() - started) * 1000), peak, tuple(vectors))
