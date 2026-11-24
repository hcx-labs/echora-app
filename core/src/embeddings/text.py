from __future__ import annotations

from core.settings import get_settings

from dataclasses import dataclass
import threading
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class LyricsEmbeddingResult:
    aggregate: np.ndarray
    windows: list[np.ndarray]
    token_ranges: list[tuple[int, int]]
    inference_ms: int
    peak_vram_bytes: int | None


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


_shared_model = None
_shared_lock = threading.Lock()


class LyricsEmbeddingModel:
    name = "bge_m3"
    dimension = 1024
    chunk_tokens = 7168
    overlap_tokens = 512

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        self.device = torch.device(device)
        self.revision = revision
        self.model = SentenceTransformer(model_id, revision=revision, device=device)
        self.model.max_seq_length = 8192
        self.tokenizer = self.model.tokenizer

    def _chunks(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError("Lyrics contain no tokens")
        chunks: list[str] = []
        ranges: list[tuple[int, int]] = []
        step = self.chunk_tokens - self.overlap_tokens
        for start in range(0, len(token_ids), step):
            end = min(len(token_ids), start + self.chunk_tokens)
            chunks.append(self.tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
            ranges.append((start, end))
            if end == len(token_ids):
                break
        return chunks, ranges

    def embed(self, text: str) -> LyricsEmbeddingResult:
        chunks, ranges = self._chunks(text)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            encoded = self.model.encode(
                chunks, batch_size=get_settings().lyrics_window_batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
        vectors = [_normalize(row) for row in encoded]
        aggregate = _normalize(np.mean(vectors, axis=0))
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        return LyricsEmbeddingResult(
            aggregate=aggregate, windows=vectors, token_ranges=ranges,
            inference_ms=round((time.perf_counter() - started) * 1000), peak_vram_bytes=peak,
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        matrix = self.model.encode(
            [text.strip() for text in texts], batch_size=get_settings().lyrics_track_batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(matrix, dtype=np.float32)


def shared_lyrics_model() -> LyricsEmbeddingModel:
    global _shared_model
    with _shared_lock:
        if _shared_model is None:
            _shared_model = LyricsEmbeddingModel(
                get_settings().lyrics_model_id,
                get_settings().lyrics_revision,
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        return _shared_model
