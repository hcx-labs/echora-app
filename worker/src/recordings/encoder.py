"""Local, opt-in recording embeddings; not a recording recognition decision.

Requires a parity-validated, self-contained ONNX waveform-input export with its
frontend embedded. An upstream TF checkpoint or direct AudioMuse graph is NOT
supported. No downloads, guessed mel frontend, resampling, or fake embeddings.
The parent supplies mono 8 kHz audio and separately gates recognition on calibrated
matcher thresholds. Deployment attestations are operator assertions, not proof of
parity; the report checksum pins the evidence reviewed by that operator.

ECHORA_RECORDING_MODEL_MANIFEST points to JSON with exactly these fields::

    {"schema_version": 1, "enabled": true, "license_acknowledged": true,
     "parity_validated": true, "model_path": "encoder.onnx",
     "model_sha256": "<64 lowercase hex>",
     "validation_report_path": "parity.json",
     "validation_report_sha256": "<64 lowercase hex>",
     "frontend": "embedded-waveform-v1", "sample_rate": 8000,
     "window_samples": 8000, "hop_samples": 4000, "embedding_dim": 128,
     "input_name": "waveform", "output_name": "embedding", "batch_size": 32}

Paths are local, relative to the manifest (absolute paths also accepted). Missing
environment variable disables this feature; present but invalid manifests raise
ValueError. load_config/config_from_env and validate_config verify files without
importing ONNX Runtime. Config is frozen, and validation detects replaced fields.
representation_id pins all representation semantics including artifact digest;
config_hash additionally pins deployment paths, attestations and batch size.

encode returns float32 [N,128], N=1+(len(samples)-8000)//4000. Only complete
windows are used, at times i*0.5 seconds. Exact silent windows remain zero rows;
non-silent rows must have nonzero finite embeddings and are L2 normalized. Inputs
shorter than one second are rejected. No padding or amplitude normalization occurs.
"""

from core.settings import get_settings

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np


_FIXED = {
    "schema_version": 1, "frontend": "embedded-waveform-v1", "sample_rate": 8000,
    "window_samples": 8000, "hop_samples": 4000, "embedding_dim": 128,
}
_GATES = ("enabled", "license_acknowledged", "parity_validated")
_PATHS = ("model_path", "validation_report_path")
_DIGESTS = ("model_sha256", "validation_report_sha256")
_FIELDS = set(_FIXED) | set(_GATES) | set(_PATHS) | set(_DIGESTS) | {
    "input_name", "output_name", "batch_size",
}


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _representation(data: dict) -> str:
    keys = set(_FIXED) | {"model_sha256", "input_name", "output_name"}
    return _hash({**{k: data[k] for k in keys}, "encoder_revision": 1,
                  "normalization": "l2-float64", "silence": "exact-zero-preserve-row",
                  "tail": "drop-incomplete", "waveform": "mono-float32-no-scaling"})


def _validate_manifest(data: dict) -> None:
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ValueError("Manifest must contain exactly the documented fields")
    for key, value in _FIXED.items():
        if type(data[key]) is not type(value) or data[key] != value:
            raise ValueError(f"Unsupported {key}")
    for key in _GATES:
        if data[key] is not True:
            raise ValueError(f"Deployment requires {key}=true")
    for key in _DIGESTS:
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise ValueError(f"Invalid {key}")
    for key in (*_PATHS, "input_name", "output_name"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"Invalid {key}")
    if type(data["batch_size"]) is not int or not 1 <= data["batch_size"] <= 256:
        raise ValueError("batch_size must be an integer in [1,256]")


_CACHE_LOCK = RLock()
_VERIFIED = OrderedDict()  # Metadata only; never retains artifact contents.
_SESSION = None  # At most one model generation per worker process.


def _signature(path: str) -> tuple:
    try:
        stat = Path(path).stat()
    except OSError as exc:
        raise ValueError(f"Cannot read local artifact: {path}") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _verify_file(path: str, digest: str, check: Callable[[], None]) -> tuple:
    signature = _signature(path)
    key = (path, digest, signature)
    with _CACHE_LOCK:
        check()
        if key in _VERIFIED:
            _VERIFIED.move_to_end(key)
            return signature
        value = hashlib.sha256()
        try:
            with Path(path).open('rb') as source:
                while block := source.read(1024 * 1024):
                    check()
                    value.update(block)
        except OSError as exc:
            raise ValueError(f"Cannot read local artifact: {path}") from exc
        if value.hexdigest() != digest or _signature(path) != signature:
            raise ValueError(f"SHA256 mismatch or artifact replaced during verification: {path}")
        _VERIFIED[key] = True
        while len(_VERIFIED) > 8:
            _VERIFIED.popitem(last=False)
        return signature


def _session(data: dict, check: Callable[[], None]):
    global _SESSION
    with _CACHE_LOCK:
        check()
        key = (_hash(data), tuple(_verify_file(data[path], data[digest], check)
                                 for path, digest in zip(_PATHS, _DIGESTS)))
        if _SESSION is not None and _SESSION[0] == key:
            return _SESSION[1]
        # Drop the old generation before allocating the next model.
        _SESSION = None
        model = _verified_bytes(data['model_path'], data['model_sha256'])
        check()
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(get_settings().recording_intra_op_threads, os.cpu_count() or 1)
        options.inter_op_num_threads = get_settings().recording_inter_op_threads
        session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
        check()
        _check_graph(session, data)
        if tuple(_signature(data[path]) for path in _PATHS) != key[1]:
            raise ValueError('Artifact replaced during session initialization')
        _SESSION = (key, session)
        return session


def _verified_bytes(path: str, digest: str) -> bytes:
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read local artifact: {path}") from exc
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"SHA256 mismatch: {path}")
    return content


@dataclass(frozen=True)
class EncoderConfig:
    """Created by load_config; canonical JSON avoids mutable nested configuration."""

    manifest_json: str
    representation_id: str
    config_hash: str

    @property
    def batch_size(self) -> int:
        return json.loads(self.manifest_json)["batch_size"]


def _config_data(config: EncoderConfig) -> dict:
    if not isinstance(config, EncoderConfig):
        raise ValueError("Expected EncoderConfig")
    data = json.loads(config.manifest_json)
    _validate_manifest(data)
    if _hash(data) != config.config_hash or _representation(data) != config.representation_id:
        raise ValueError("Configuration hash mismatch")
    if any(not Path(data[key]).is_absolute() for key in _PATHS):
        raise ValueError("Configuration artifact paths must be absolute")
    return data


def validate_config(config: EncoderConfig, check: Callable[[], None] = lambda: None) -> bool:
    """Verify configuration and pinned artifacts without loading ORT; raise on failure."""
    data = _config_data(config)
    for path, digest in zip(_PATHS, _DIGESTS):
        _verify_file(data[path], data[digest], check)
    return True


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def load_config(manifest_path: str | Path) -> EncoderConfig:
    """Read a strict manifest and verify its model and parity report checksums."""
    path = Path(manifest_path).resolve()
    try:
        data = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError) as exc:
        raise ValueError("Cannot read model manifest") from exc
    _validate_manifest(data)
    for key in _PATHS:
        data[key] = str((path.parent / data[key]).resolve())
    config = EncoderConfig(json.dumps(data, sort_keys=True), _representation(data), _hash(data))
    validate_config(config)
    return config


def config_from_env(environ: Mapping[str, str] | None = None) -> EncoderConfig | None:
    """Absent/empty env disables; an invalid configured path fails closed."""

    path = (get_settings().recording_model_manifest if environ is None
            else environ.get("ECHORA_RECORDING_MODEL_MANIFEST"))
    if path is None or not path.strip():
        return None
    return load_config(path)


def _check_graph(session, data: dict) -> None:
    for nodes, name, width in (
        (session.get_inputs(), data["input_name"], 8000),
        (session.get_outputs(), data["output_name"], 128),
    ):
        if len(nodes) != 1:
            raise ValueError("Graph must have exactly one input and one output")
        node = nodes[0]
        shape = node.shape
        if (node.name != name or node.type != "tensor(float)" or len(shape) != 2
                or shape[1] != width or not (shape[0] is None or isinstance(shape[0], str))):
            raise ValueError("Graph requires dynamic-batch float32 waveform [B,8000] -> [B,128]")


def encode(samples: np.ndarray, config: EncoderConfig,
           check: Callable[[], None] = lambda: None) -> np.ndarray:
    """Run bounded CPU batches. check() may raise to cancel before/between ORT calls."""
    check()
    data = _config_data(config)
    if (not isinstance(samples, np.ndarray) or samples.dtype != np.float32
            or samples.ndim != 1 or samples.size < 8000 or not np.isfinite(samples).all()):
        raise ValueError("Expected finite mono float32 8 kHz samples, at least 8000 samples")
    validate_config(config, check)
    session = None
    count = 1 + (samples.size - 8000) // 4000
    result = np.zeros((count, 128), dtype=np.float32)
    for start in range(0, count, data["batch_size"]):
        check()
        stop = min(count, start + data["batch_size"])
        batch = np.stack([samples[i * 4000:i * 4000 + 8000] for i in range(start, stop)])
        active = np.any(batch != 0, axis=1)
        if not active.any():
            continue
        if session is None:
            # Loading verified bytes avoids unpinned external tensor files. Silent
            # inputs still validate configuration, but never initialize ORT.
            session = _session(data, check)
        output = session.run([data["output_name"]], {data["input_name"]: batch[active]})[0]
        check()
        if (not isinstance(output, np.ndarray) or output.dtype != np.float32
                or output.shape != (int(active.sum()), 128) or not np.isfinite(output).all()):
            raise ValueError("Invalid encoder output")
        values = output.astype(np.float64)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero embedding for non-silent waveform")
        result[start:stop][active] = (values / norms).astype(np.float32)
    check()
    return result
