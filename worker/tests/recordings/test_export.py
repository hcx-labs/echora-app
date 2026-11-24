"""Export provenance checks require no model runtime or network access."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

_PATH = Path(__file__).resolve().parents[3] / "scripts/export_recording_model.py"
_spec = importlib.util.spec_from_file_location("recording_export", _PATH)
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    files = [folder / "config.yaml", folder / "ckpt-100.index", folder / "ckpt-100.data-00000-of-00001"]
    bundle = tmp_path / "model.zip"
    with zipfile.ZipFile(bundle, "w") as writer:
        for index, path in enumerate(files):
            path.write_bytes(f"fixture-{index}".encode())
            writer.write(path, "nmfp-triplet/" + path.name)
    monkeypatch.setattr(exporter, "CHECKPOINT_ARCHIVE_MD5", hashlib.md5(bundle.read_bytes()).hexdigest())
    return bundle, folder, files


def test_archive_and_each_checkpoint_file_must_match(archive):
    bundle, folder, files = archive
    assert exporter.verify_checkpoint_archive(bundle, folder, files) == exporter.sha256(bundle)
    files[0].write_text("modified")
    with pytest.raises(ValueError, match="differs"):
        exporter.verify_checkpoint_archive(bundle, folder, files)


def test_wrong_archive_cannot_claim_published_provenance(archive, monkeypatch):
    bundle, folder, files = archive
    monkeypatch.setattr(exporter, "CHECKPOINT_ARCHIVE_MD5", "0" * 32)
    with pytest.raises(ValueError, match="published checksum"):
        exporter.verify_checkpoint_archive(bundle, folder, files)


def test_missing_archive_member_fails_closed(archive):
    bundle, folder, files = archive
    missing = folder / "unpublished.index"
    missing.write_text("unpublished")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        exporter.verify_checkpoint_archive(bundle, folder, [*files, missing])


def test_validation_tolerances_and_pins_are_explicit():
    assert len(exporter.SOURCE_COMMIT) == 40
    assert exporter.FRONTEND_MAX_ERROR <= .005
    assert exporter.EMBEDDING_MAX_ERROR <= .0005
    assert exporter.EMBEDDING_MIN_COSINE >= .99999
