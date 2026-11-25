"""Shared, content-addressed audio prerequisites for analysis jobs.

Only derived audio lives here. Source authorization/downloads remain with callers.
Entries are immutable by source bytes + recipe and published atomically under a
process lock. Consumers receive independent arrays, never mutable shared buffers.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np

from worker.media.config import PreprocessConfig

CACHE_SCHEMA = "audio-prerequisites-v1"
DECODE_REVISION = "ffmpeg-f32le-v1"
_state: ContextVar = ContextVar("audio_preprocessing", default=None)


def get_check():
    state = _state.get()
    return state[1] if state else lambda: None


class AudioArtifactCache:
    def __init__(self, directory=None, *, max_bytes=None, ttl_seconds=None, check=None):
        self.directory = Path(directory or PreprocessConfig.load().dir)
        self.max_bytes = int(
            max_bytes if max_bytes is not None else PreprocessConfig.load().max_bytes
        )
        self.ttl_seconds = float(
            ttl_seconds if ttl_seconds is not None else PreprocessConfig.load().ttl_seconds
        )
        if self.max_bytes < 0 or self.ttl_seconds <= 0:
            raise ValueError("Preprocessing cache needs nonnegative max bytes and positive TTL")
        self.check = check or get_check()

    @staticmethod
    def key(source: bytes, recipe: dict) -> tuple[str, dict]:
        metadata = {
            "schema": CACHE_SCHEMA,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "recipe": recipe,
        }
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return digest, metadata

    @contextmanager
    def _lock(self, key, *, blocking=True):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Lock files are stable: unlinking a held lock would allow two writers.
        with (self.directory / f"{key}.lock").open("a+b") as lock:
            deadline = time.monotonic() + PreprocessConfig.load().lock_timeout_seconds
            acquired = False
            try:
                while not acquired:
                    self.check()
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except BlockingIOError:
                        if not blocking:
                            yield False
                            return
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Timed out waiting for an audio prerequisite")
                        threading.Event().wait(PreprocessConfig.load().lock_poll_seconds)
                yield True
            finally:
                if acquired:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _load(self, key, expected):
        path = self.directory / f"{key}.npy"
        manifest = self.directory / f"{key}.json"
        try:
            if time.time() - path.stat().st_mtime > self.ttl_seconds:
                return None
            metadata = json.loads(manifest.read_text())
            if not isinstance(metadata, dict):
                return None
            if any(metadata.get(k) != v for k, v in expected.items()):
                return None
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != metadata.get("sha256"):
                return None
            array = np.load(io.BytesIO(content), allow_pickle=False)
            if not isinstance(array, np.ndarray):
                array.close()
                return None
            if array.dtype != np.float32 or list(array.shape) != metadata.get("shape"):
                return None
            if not array.size or not np.isfinite(array).all():
                return None
            os.utime(path, None)
            return array
        except (OSError, ValueError, TypeError, KeyError, EOFError):
            return None

    def array(self, source: bytes, recipe: dict, produce):
        self.check()
        key, metadata = self.key(source, recipe)
        with self._lock(key):
            result = self._load(key, metadata)
            if result is not None:
                self.check()
                return result
            result = np.asarray(produce(), dtype=np.float32)
            self.check()
            if result.ndim not in (1, 2) or not result.size or not np.isfinite(result).all():
                raise ValueError("Invalid preprocessed audio")
            output = io.BytesIO()
            np.save(output, result, allow_pickle=False)
            content = output.getvalue()
            metadata.update(shape=list(result.shape), sha256=hashlib.sha256(content).hexdigest())
            encoded = json.dumps(metadata, sort_keys=True).encode()
            if len(content) + len(encoded) <= self.max_bytes:
                # Both temporaries are private. The manifest is the commit marker.
                paths = []
                try:
                    for suffix, payload in (("npy", content), ("json", encoded)):
                        with tempfile.NamedTemporaryFile(
                            dir=self.directory, prefix=f".{key}.", suffix=".tmp", delete=False
                        ) as stream:
                            paths.append(Path(stream.name))
                            stream.write(payload)
                            stream.flush()
                            os.fsync(stream.fileno())
                        self.check()
                        os.replace(paths[-1], self.directory / f"{key}.{suffix}")
                finally:
                    for path in paths:
                        path.unlink(missing_ok=True)
        self.prune()
        return result.copy()

    def prune(self):
        """Best-effort LRU/TTL eviction; skip entries actively read or produced."""
        if not self.directory.exists():
            return
        with self._lock("maintenance", blocking=False) as acquired:
            if not acquired:
                return
            entries = []
            now = time.time()
            for path in self.directory.glob("*.npy"):
                try:
                    stat = path.stat()
                    manifest = path.with_suffix(".json")
                    size = stat.st_size + (manifest.stat().st_size if manifest.exists() else 0)
                    entries.append((stat.st_mtime, size, path))
                except FileNotFoundError:
                    continue
            total = sum(item[1] for item in entries)
            for modified, size, path in sorted(entries):
                self.check()
                if total <= self.max_bytes and now - modified <= self.ttl_seconds:
                    # Entries are oldest-first: every remaining entry is also
                    # within TTL and budget. Avoid a database-backed cancellation
                    # check per retained artifact on every cache publication.
                    break
                with self._lock(path.stem, blocking=False) as free:
                    if not free:
                        continue
                    # A reader may have touched the entry since the directory scan.
                    try:
                        if (
                            total <= self.max_bytes
                            and now - path.stat().st_mtime <= self.ttl_seconds
                        ):
                            continue
                    except FileNotFoundError:
                        continue
                    path.unlink(missing_ok=True)
                    path.with_suffix(".json").unlink(missing_ok=True)
                    total -= size
            # Hard-killed writers can leave private staging files; recover them.
            for path in self.directory.glob(".*.tmp"):
                try:
                    if now - path.stat().st_mtime <= 86400:
                        continue
                    key = path.name.split(".")[1]
                    with self._lock(key, blocking=False) as free:
                        if free:
                            path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass


@contextmanager
def preprocessing_session(check=lambda: None):
    cache = AudioArtifactCache(check=check)
    token = _state.set((cache, check))
    try:
        yield cache
    finally:
        _state.reset(token)


def active_cache():
    state = _state.get()
    return state[0] if state else None


def _cache(check):
    # Compose explicit caller and job cancellation rather than losing either one.
    current = get_check()

    def combined():
        current()
        if check is not current:
            check()

    return AudioArtifactCache(check=combined)


def cached_decode(data, sample_rate, channels, produce):
    cache = active_cache()
    if cache is None:
        return produce()
    return cache.array(
        data,
        {
            "kind": "decode",
            "revision": DECODE_REVISION,
            "sample_rate": sample_rate,
            "channels": channels,
        },
        produce,
    )


def _vocal_recipe():
    from worker.media.roformer import SEPARATION_REVISION

    return {
        "kind": "vocals",
        "separator": SEPARATION_REVISION,
        "sample_rate": 44100,
        "channels": 2,
        "decode_revision": DECODE_REVISION,
    }


def vocal_stereo(data: bytes, check=lambda: None, *, before_separate=None):
    from worker.media.audio import decode_audio_channels
    from worker.media.roformer import separate_vocals

    cache = _cache(check)
    # Dependencies acquire distinct keys in mono -> stereo -> decode order.
    # On a cache hit, no decoding or GPU model loading is necessary.
    recipe = _vocal_recipe()

    def produce():
        channels = decode_audio_channels(data, 44100, 2)
        cache.check()
        if before_separate is not None:
            before_separate()
        return separate_vocals(channels, check=cache.check)

    return cache.array(data, recipe, produce)


def vocal_waveform(data: bytes, sample_rate=16000, check=lambda: None, *, before_separate=None):
    import soxr

    cache = _cache(check)
    recipe = {
        **_vocal_recipe(),
        "kind": "vocals-mono",
        "sample_rate": int(sample_rate),
        "channels": 1,
        "resampler": f"soxr-{soxr.__version__}-HQ",
    }

    def produce():
        vocals = vocal_stereo(data, check, before_separate=before_separate).mean(axis=1)
        if sample_rate != 44100:
            vocals = soxr.resample(vocals, 44100, sample_rate, quality="HQ")
        return np.asarray(vocals, dtype=np.float32)

    return cache.array(data, recipe, produce)


def vocal_reference_waveform(data: bytes, check=lambda: None):
    """Original mix on exactly the same sample grid as mono16k Roformer vocals."""
    import soxr

    from worker.media.audio import decode_audio_channels

    recipe = {
        "kind": "vocal-reference-mono",
        "revision": DECODE_REVISION,
        "sample_rate": 16000,
        "channels": 1,
        "input_rate": 44100,
        "resampler": f"soxr-{soxr.__version__}-HQ",
        "downmix": "stereo-mean-v1",
    }

    def produce():
        mix = decode_audio_channels(data, 44100, 2).mean(axis=1)
        return np.asarray(soxr.resample(mix, 44100, 16000, quality="HQ"), dtype=np.float32)

    return _cache(check).array(data, recipe, produce)


def melody_waveforms(data: bytes, check=lambda: None):
    """Mono 44.1 kHz vocals and mix-minus-vocals, from the same cached stereo input."""
    from worker.media.audio import decode_audio_channels

    cache = _cache(check)
    vocals = vocal_waveform(data, 44100, check)
    recipe = {
        **_vocal_recipe(),
        "kind": "accompaniment-mono",
        "channels": 1,
        "derivation": "stereo-mix-minus-vocals-then-mean-v1",
    }

    def produce():
        stem = vocal_stereo(data, check)
        mix = decode_audio_channels(data, 44100, 2)
        if stem.shape != mix.shape:
            raise ValueError("Vocal stem does not match the decoded source shape")
        return (mix - stem).mean(axis=1, dtype=np.float32)

    accompaniment = cache.array(data, recipe, produce)
    if vocals.shape != accompaniment.shape:
        raise ValueError("Melody sources differ in length")
    return vocals, accompaniment


def prepare_audio(
    data: bytes,
    *,
    mono_rates=(),
    stereo_rates=(),
    stereo=False,
    vocals=False,
    melody=False,
    reference=False,
    check=lambda: None,
):
    from worker.media.audio import decode_audio, decode_audio_channels

    check()
    for sample_rate in sorted(set(mono_rates)):
        check()
        decode_audio(data, sample_rate)
    for sample_rate in sorted(set(stereo_rates) | ({44100} if stereo else set())):
        check()
        decode_audio_channels(data, sample_rate, 2)
    if vocals:
        vocal_waveform(data, 16000, check)
    if melody:
        melody_waveforms(data, check)
    if reference:
        vocal_reference_waveform(data, check)
    check()
