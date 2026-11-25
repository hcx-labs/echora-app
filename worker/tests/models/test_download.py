import pytest

from core.config import reset_config_cache


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    reset_config_cache()
    yield
    reset_config_cache()


from unittest.mock import Mock

from worker.models.download import _pin_main_ref


def test_pin_main_ref_points_to_snapshot_commit(tmp_path) -> None:
    snapshot = tmp_path / "models--example" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    _pin_main_ref(str(snapshot))

    assert (tmp_path / "models--example" / "refs" / "main").read_text() == "abc123"


def test_required_models_include_pinned_roformer(monkeypatch):
    from worker.media.roformer import MODEL_ID, MODEL_REVISION
    from worker.models.download import required_models

    monkeypatch.delenv("MOSS_MODEL_ID", raising=False)
    monkeypatch.delenv("MOSS_REVISION", raising=False)
    assert (MODEL_ID, MODEL_REVISION, False) in required_models()


def test_provisioning_leaves_legacy_demucs_weights_untouched(monkeypatch, tmp_path):
    import worker.models.download as d

    checkpoint = tmp_path / "torch/hub/checkpoints/955717e8-8726e21a.th"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserve existing experiment weights")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setattr(
        d,
        "required_models",
        Mock(return_value=((d.ROFORMER_MODEL_ID, d.ROFORMER_REVISION, False),)),
    )
    download = Mock()
    monkeypatch.setattr(d, "snapshot_download", download)
    monkeypatch.setattr(d, "_prune_huggingface_cache", Mock())
    monkeypatch.setattr(d, "_download_essentia", Mock())
    monkeypatch.setattr(d, "download_recording_model", Mock())
    d.main()
    download.assert_called_once_with(repo_id=d.ROFORMER_MODEL_ID, revision=d.ROFORMER_REVISION)
    assert checkpoint.read_bytes() == b"preserve existing experiment weights"


def test_normal_downloader_provisions_recording_bundle(monkeypatch):
    import worker.models.download as d

    monkeypatch.setattr(d, "required_models", lambda: ())
    recording = Mock()
    monkeypatch.setattr(d, "download_recording_model", recording)
    monkeypatch.setattr(d, "_prune_huggingface_cache", Mock())
    monkeypatch.setattr(d, "_download_essentia", Mock())
    d.main()
    recording.assert_called_once_with()


def test_prune_only_never_downloads_or_validates_recording_bundle(monkeypatch):
    import worker.models.download as d

    monkeypatch.setattr(d, "required_models", lambda: ())
    recording, snapshots, essentia = Mock(), Mock(), Mock()
    monkeypatch.setattr(d, "download_recording_model", recording)
    monkeypatch.setattr(d, "snapshot_download", snapshots)
    monkeypatch.setattr(d, "_download_essentia", essentia)
    monkeypatch.setattr(d, "_prune_huggingface_cache", Mock())
    d.main(prune_only=True)
    recording.assert_not_called()
    snapshots.assert_not_called()
    essentia.assert_not_called()
