"""Local pinned dummy artifacts and fake ORT; no production fake encoder."""

import hashlib
import json
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

import worker.recordings.encoder as encoder


def digest(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def manifest(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"local graph")
    (tmp_path / "parity.json").write_bytes(b"reviewed parity evidence")
    data = {
        "schema_version": 1,
        "enabled": True,
        "license_acknowledged": True,
        "parity_validated": True,
        "model_path": "model.onnx",
        "model_sha256": digest(b"local graph"),
        "validation_report_path": "parity.json",
        "validation_report_sha256": digest(b"reviewed parity evidence"),
        "frontend": "embedded-waveform-v1",
        "sample_rate": 8000,
        "window_samples": 8000,
        "hop_samples": 4000,
        "embedding_dim": 128,
        "input_name": "waveform",
        "output_name": "embedding",
        "batch_size": 2,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data


@pytest.fixture
def ort(monkeypatch):
    state = SimpleNamespace(calls=[], loads=[], bad_output=None, bad_shape=False)

    class Session:
        def __init__(self, model, providers, sess_options):
            assert 1 <= sess_options.intra_op_num_threads <= 4
            assert sess_options.inter_op_num_threads == 1
            assert model == b"local graph"
            assert providers == ["CPUExecutionProvider"]
            state.loads.append(model)

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="waveform",
                    type="tensor(float)",
                    shape=["batch", 7999 if state.bad_shape else 8000],
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="embedding", type="tensor(float)", shape=[None, 128])]

        def run(self, names, feeds):
            assert names == ["embedding"]
            batch = feeds["waveform"]
            assert batch.dtype == np.float32
            state.calls.append(batch.copy())
            if state.bad_output is not None:
                return [state.bad_output]
            output = np.ones((len(batch), 128), dtype=np.float32)
            output[:, 0] = batch[:, 0]
            return [output]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=Session, SessionOptions=SimpleNamespace),
    )
    return state


def test_metadata_without_ort(manifest, monkeypatch):
    path, _ = manifest
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert encoder.config_from_env({}) is None
    assert encoder.config_from_env({"ECHORA_RECORDING_MODEL_MANIFEST": ""}) is None
    assert encoder.config_from_env({"ECHORA_RECORDING_MODEL_MANIFEST": "  "}) is None
    config = encoder.config_from_env({"ECHORA_RECORDING_MODEL_MANIFEST": str(path)})
    assert encoder.validate_config(config)
    assert len(config.representation_id) == 64
    assert encoder.load_config(path) == config
    with pytest.raises(FrozenInstanceError):
        config.config_hash = "changed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", False),
        ("license_acknowledged", False),
        ("parity_validated", False),
        ("enabled", 1),
        ("sample_rate", 16000),
        ("window_samples", 7999),
        ("hop_samples", 8000),
        ("embedding_dim", 64),
        ("frontend", "mel"),
        ("batch_size", 0),
        ("batch_size", 257),
        ("batch_size", True),
        ("schema_version", True),
        ("model_sha256", "bad"),
        ("input_name", ""),
        ("unexpected", True),
    ],
)
def test_manifest_rejected(manifest, field, value):
    path, data = manifest
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        encoder.load_config(path)


@pytest.mark.parametrize("filename", ["model.onnx", "parity.json"])
def test_digest_checked_before_load(manifest, ort, filename):
    path, _ = manifest
    config = encoder.load_config(path)
    (path.parent / filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        encoder.load_config(path)
    with pytest.raises(ValueError, match="SHA256"):
        encoder.encode(np.ones(8000, np.float32), config)
    assert not ort.loads


def test_hash_consistency(manifest):
    path, data = manifest
    config = encoder.load_config(path)
    data["batch_size"] = 3
    path.write_text(json.dumps(data))
    other = encoder.load_config(path)
    assert config.config_hash != other.config_hash
    assert config.representation_id == other.representation_id
    with pytest.raises(ValueError, match="hash"):
        encoder.validate_config(replace(config, manifest_json=other.manifest_json))
    data["input_name"] = "another_input"
    path.write_text(json.dumps(data))
    assert encoder.load_config(path).representation_id != config.representation_id
    with pytest.raises(ValueError, match="hash"):
        encoder.validate_config(replace(config, representation_id="0" * 64))


def test_batching_timing_silence_and_tail(manifest, ort):
    config = encoder.load_config(manifest[0])
    audio = np.arange(28017, dtype=np.float32)
    audio[8000:20000] = 0
    checks = []
    result = encoder.encode(audio, config, check=lambda: checks.append(True))
    assert result.shape == (6, 128)
    assert result.dtype == np.float32
    assert all(len(batch) <= 2 for batch in ort.calls)
    expected = np.stack([audio[i * 4000 : i * 4000 + 8000] for i in range(6)])
    active = np.any(expected != 0, axis=1)
    np.testing.assert_array_equal(np.concatenate(ort.calls), expected[active])
    np.testing.assert_array_equal(result[~active], 0)
    np.testing.assert_allclose(np.linalg.norm(result[active], axis=1), 1, atol=1e-6)
    assert len(checks) >= 6


def test_silence(manifest, ort):
    result = encoder.encode(np.zeros(12000, np.float32), encoder.load_config(manifest[0]))
    np.testing.assert_array_equal(result, np.zeros((2, 128), np.float32))
    assert not ort.calls


@pytest.mark.parametrize(
    "audio",
    [
        np.ones(7999, np.float32),
        np.ones(8000, np.float64),
        np.ones((8000, 1), np.float32),
        np.full(8000, np.nan, np.float32),
        np.full(8000, np.inf, np.float32),
        [0] * 8000,
    ],
)
def test_invalid_audio(manifest, ort, audio):
    with pytest.raises(ValueError, match="finite mono"):
        encoder.encode(audio, encoder.load_config(manifest[0]))
    assert not ort.loads


def test_graph_shape(manifest, ort):
    ort.bad_shape = True
    with pytest.raises(ValueError, match="dynamic-batch"):
        encoder.encode(np.ones(8000, np.float32), encoder.load_config(manifest[0]))
    assert not ort.calls


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 128), np.float32),
        np.full((1, 128), np.nan, np.float32),
        np.ones((1, 127), np.float32),
        np.ones((1, 128), np.float64),
    ],
)
def test_bad_outputs(manifest, ort, output):
    ort.bad_output = output
    with pytest.raises(ValueError):
        encoder.encode(np.ones(8000, np.float32), encoder.load_config(manifest[0]))


def test_cancel(manifest, ort):
    config = encoder.load_config(manifest[0])

    def cancel():
        if ort.calls:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        encoder.encode(np.ones(32000, np.float32), config, check=cancel)
    assert len(ort.calls) == 1


def test_missing_and_duplicate_fields(manifest):
    path, data = manifest
    del data["parity_validated"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        encoder.load_config(path)
    path.write_text('{"enabled":true,"enabled":false}')
    with pytest.raises(ValueError, match="Duplicate"):
        encoder.load_config(path)


def test_repeated_validation_does_not_reread_artifacts(manifest, monkeypatch):
    from pathlib import Path

    path, _ = manifest
    config = encoder.load_config(path)
    original = Path.open
    reads = []

    def opened(file, *args, **kwargs):
        reads.append(file.name)
        return original(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    encoder.load_config(path)
    encoder.validate_config(config)
    assert reads == ["manifest.json"]
    assert len(encoder._VERIFIED) <= 8


def test_session_reused_and_silence_never_loads_it(manifest, ort):
    path, _ = manifest
    config = encoder.load_config(path)
    encoder.encode(np.zeros(8000, np.float32), config)
    assert ort.loads == []
    encoder.encode(np.ones(8000, np.float32), config)
    encoder.encode(np.ones(8000, np.float32), config)
    assert len(ort.loads) == 1


@pytest.mark.parametrize("filename", ["model.onnx", "parity.json"])
def test_same_path_replacement_invalidates_cached_session(manifest, ort, filename):
    path, _ = manifest
    config = encoder.load_config(path)
    samples = np.ones(8000, np.float32)
    encoder.encode(samples, config)
    artifact = path.parent / filename
    replacement = path.parent / "replacement"
    replacement.write_bytes(artifact.read_bytes())
    replacement.replace(artifact)
    encoder.encode(samples, config)
    assert len(ort.loads) == 2
    artifact.write_bytes(b"x" * artifact.stat().st_size)
    with pytest.raises(ValueError, match="SHA256"):
        encoder.encode(samples, config)
    assert len(ort.loads) == 2


def test_session_cache_keeps_only_current_configuration(manifest, ort):
    path, _ = manifest
    first = encoder.load_config(path)
    data = json.loads(first.manifest_json)
    data["batch_size"] = 3
    second = encoder.EncoderConfig(
        json.dumps(data), encoder._representation(data), encoder._hash(data)
    )
    for config in [first, second, first]:
        encoder.encode(np.ones(8000, np.float32), config)
        assert encoder._SESSION[0][0] == config.config_hash
    assert len(ort.loads) == 3
