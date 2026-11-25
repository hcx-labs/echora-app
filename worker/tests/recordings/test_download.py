"""Provision dummy pinned artifacts without network or inference."""

import hashlib
import json
import shutil
from dataclasses import asdict
from unittest.mock import Mock

import pytest

import worker.recordings.download as downloader
from core.config import reset_config_cache
from worker.recordings.encoder import load_config
from worker.recordings.matcher import MATCHER_REVISION, MatchPolicy

REVISION = "a" * 40


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "encoder.onnx").write_bytes(b"unit-test artifact, not an inference model")
    (source / "parity.json").write_text('{"passed":true}')
    manifest = {
        "schema_version": 1,
        "enabled": True,
        "license_acknowledged": True,
        "parity_validated": True,
        "model_path": "encoder.onnx",
        "model_sha256": hashlib.sha256((source / "encoder.onnx").read_bytes()).hexdigest(),
        "validation_report_path": "parity.json",
        "validation_report_sha256": hashlib.sha256(
            (source / "parity.json").read_bytes()
        ).hexdigest(),
        "frontend": "embedded-waveform-v1",
        "sample_rate": 8000,
        "window_samples": 8000,
        "hop_samples": 4000,
        "embedding_dim": 128,
        "input_name": "waveform",
        "output_name": "embedding",
        "batch_size": 32,
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    config = load_config(source / "manifest.json")
    (source / "match-policy.json").write_text(
        json.dumps(
            {
                "representation_id": config.representation_id,
                "matcher_revision": MATCHER_REVISION,
                "calibrated": True,
                "validation_dataset": "unit-test fixtures only",
                "thresholds": asdict(MatchPolicy(0.5, 0.6, 6, 0.5, 0.5, 0.03)),
            }
        )
    )
    for name in ["UPSTREAM-LICENSE", "README.md"]:
        (source / name).write_text("unit-test fixture")
    return source


@pytest.fixture
def provision(monkeypatch, tmp_path, bundle):
    root = tmp_path / "installed"
    env = {
        "ECHORA_RECORDING_MODEL_ID": "example/recording",
        "ECHORA_RECORDING_MODEL_REVISION": REVISION,
        "ECHORA_RECORDING_MODEL_DIRECTORY": str(root),
    }

    def copy(**kwargs):
        assert kwargs["repo_id"] == env["ECHORA_RECORDING_MODEL_ID"]
        assert kwargs["revision"] == REVISION
        assert kwargs["allow_patterns"] == list(downloader.BUNDLE_FILES)
        assert not (root / REVISION).exists()
        shutil.copytree(bundle, kwargs["local_dir"], dirs_exist_ok=True)

    fetch = Mock(side_effect=copy)
    monkeypatch.setattr(downloader, "snapshot_download", fetch)
    return env, fetch, root


def test_disabled_is_noop(provision):
    _, fetch, root = provision
    assert downloader.download_recording_model({}) is None
    assert not root.exists()
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"ECHORA_RECORDING_MODEL_ID": ""},
        {"ECHORA_RECORDING_MODEL_REVISION": ""},
        {"ECHORA_RECORDING_MODEL_REVISION": "main"},
        {"ECHORA_RECORDING_MODEL_REVISION": "a" * 39},
        {"ECHORA_RECORDING_MODEL_DIRECTORY": "relative/path"},
    ],
)
def test_invalid_config_fails_before_download(provision, change):
    env, fetch, _ = provision
    with pytest.raises(ValueError):
        downloader.download_recording_model({**env, **change})
    fetch.assert_not_called()


def test_cold_install_and_offline_reuse_preserve_other_generations(provision, monkeypatch):
    env, fetch, root = provision
    sibling = root / ("b" * 40)
    sibling.mkdir(parents=True)
    (sibling / "keep").write_text("previous deployment")
    monkeypatch.setenv("ECHORA_RECORDING_MODEL_MANIFEST", "/unrelated/manifest.json")
    reset_config_cache()
    installed = downloader.download_recording_model(env)
    assert installed == root / REVISION
    assert installed.stat().st_mode & 0o777 == 0o755
    for name in downloader.BUNDLE_FILES:
        assert (installed / name).stat().st_mode & 0o777 == 0o644
    assert not list(root.glob(".recording-download-*"))
    assert (sibling / "keep").read_text() == "previous deployment"
    fetch.side_effect = AssertionError("warm cache must not contact the network")
    assert downloader.download_recording_model(env) == installed
    assert fetch.call_count == 1
    import os

    assert os.environ["ECHORA_RECORDING_MODEL_MANIFEST"] == "/unrelated/manifest.json"


@pytest.mark.parametrize(
    "file", ["encoder.onnx", "parity.json", "match-policy.json", "UPSTREAM-LICENSE"]
)
def test_invalid_download_is_not_published(provision, bundle, file):
    env, _, root = provision
    if file == "UPSTREAM-LICENSE":
        (bundle / file).unlink()
    elif file == "match-policy.json":
        data = json.loads((bundle / file).read_text())
        data["matcher_revision"] = "incompatible"
        (bundle / file).write_text(json.dumps(data))
    else:
        (bundle / file).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        downloader.download_recording_model(env)
    assert not (root / REVISION).exists()
    assert not list(root.glob(".recording-download-*"))


def test_corrupt_cache_fails_without_overwriting_it(provision):
    env, fetch, _ = provision
    installed = downloader.download_recording_model(env)
    (installed / "encoder.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        downloader.download_recording_model(env)
    assert fetch.call_count == 1
    assert (installed / "encoder.onnx").read_bytes() == b"corrupt"


def test_interrupted_download_cleans_staging(provision):
    env, fetch, root = provision
    fetch.side_effect = OSError("interrupted transfer")
    with pytest.raises(OSError):
        downloader.download_recording_model(env)
    assert list(root.iterdir()) == []


def test_concurrent_completed_install_is_validated_and_reused(provision, bundle):
    env, fetch, root = provision

    def concurrent(**kwargs):
        shutil.copytree(bundle, kwargs["local_dir"], dirs_exist_ok=True)
        shutil.copytree(bundle, root / REVISION)

    fetch.side_effect = concurrent
    assert downloader.download_recording_model(env) == root / REVISION
    assert not list(root.glob(".recording-download-*"))


def test_bundle_manifest_cannot_reference_external_model(provision, bundle):
    env, _, root = provision
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["model_path"] = str(bundle / "encoder.onnx")
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="own model"):
        downloader.download_recording_model(env)
    assert not (root / REVISION).exists()
