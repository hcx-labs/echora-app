"""Opt-in checks against the actual exported graph, never a fake ORT session.

ECHORA_TEST_RECORDING_MANIFEST=/absolute/path/manifest.json pytest <this file>
No model downloads or production configuration changes are performed.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import worker.recordings.encoder as recording_encoder


@pytest.fixture
def bundle():
    path = os.getenv("ECHORA_TEST_RECORDING_MANIFEST")
    if not path:
        pytest.skip("ECHORA_TEST_RECORDING_MANIFEST is required for real-model tests")
    pytest.importorskip("onnxruntime")
    path = Path(path).resolve()
    config = recording_encoder.load_config(path)
    return path, config


def test_real_graph_returns_normalized_embeddings_and_silent_timeline(bundle):
    _, config = bundle
    rng = np.random.default_rng(81)
    audio = np.concatenate([np.zeros(12000, dtype=np.float32),
                            rng.normal(0, .1, 20000).astype(np.float32)])
    result = recording_encoder.encode(audio, config)
    assert result.shape == (7, 128)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert not result[:2].any()
    assert np.allclose(np.linalg.norm(result[2:], axis=1), 1, atol=1e-6)


def test_real_dynamic_batch_is_independent_of_batch_size(bundle, tmp_path):
    path, config = bundle
    data = json.loads(path.read_text())
    data["batch_size"] = 1
    for key in ("model_path", "validation_report_path"):
        data[key] = str((path.parent / data[key]).resolve())
    manifest = tmp_path / "single-window.json"
    manifest.write_text(json.dumps(data))
    single = recording_encoder.load_config(manifest)
    assert single.representation_id == config.representation_id
    audio = np.random.default_rng(92).normal(0, .1, 144000).astype(np.float32)
    default_result = recording_encoder.encode(audio, config)
    single_result = recording_encoder.encode(audio, single)
    assert len(default_result) == 35  # Cross the default 32-window batch boundary.
    np.testing.assert_allclose(default_result, single_result, atol=1e-5, rtol=1e-4)


def test_preserved_exporter_and_checkpoint_provenance(bundle):
    path, _ = bundle
    manifest = json.loads(path.read_text())
    report = json.loads((path.parent / manifest["validation_report_path"]).read_text())
    assert report["passed"] is True
    assert report["source_commit"] == "15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc"
    assert report["checkpoint_archive_sha256"] == "33e6059c0bf3ee3bc1df0479cefeb92dc85f4aa2a66833d430ce771aa0d7dceb"
    assert hashlib.sha256((path.parent / "exporter.py").read_bytes()).hexdigest() == report["export_script_sha256"]
    assert all(case["same_top_track_and_offset"] for case in report["retrieval_parity"])
