from core.errors import ConfigurationError
from core.config import reset_config_cache

import pytest
from worker.transcription.song import parse_window, windows
from worker.transcription.config import transcription_model


def test_windows_cover_full_song_with_overlap():
    ranges = list(windows(221.55))
    assert ranges == [(0, 60), (48, 108), (96, 156), (144, 204), (192, 221.55)]
    assert list(windows(12)) == [(0, 12)]


def test_keep_multi_and_original_overlap():
    lines = parse_window("[1.00][S01]hello[3.00][2.00][MULTI]world[4.00]", 48, 60)
    assert lines[0]["start_ms"] == 49000
    assert lines[1]["speaker"] == "MULTI"
    assert lines[1]["start_ms"] < lines[0]["end_ms"]


@pytest.mark.parametrize(
    "text",
    [
        "[1.00][S01]unfinished",
        "[2.00][S01]bad[1.00]",
        "[0.00][S01]bad[61.00]",
        "[0.00][UNKNOWN]bad[1.00]",
        "",
    ],
)
def test_invalid_generation_is_not_published(text):
    with pytest.raises(ValueError):
        parse_window(text, 0, 60)


def test_model_config_requires_pinned_revision(monkeypatch):
    monkeypatch.delenv("MOSS_MODEL_ID", raising=False)
    reset_config_cache()
    monkeypatch.delenv("MOSS_REVISION", raising=False)
    reset_config_cache()
    assert transcription_model() is None
    monkeypatch.setenv("MOSS_MODEL_ID", "example/model")
    reset_config_cache()
    with pytest.raises(ConfigurationError):
        transcription_model()
    monkeypatch.setenv("MOSS_REVISION", "a" * 40)
    reset_config_cache()
    assert transcription_model() == ("example/model", "a" * 40)


def test_transcription_uses_shared_vocals_before_loading_asr(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import huggingface_hub
    import transformers
    import worker.transcription.song as st
    from worker.transcription.vendor.moss.processor import MossTranscribeDiarizeProcessor

    class PreparedBoundary(Exception):
        pass

    monkeypatch.setattr(huggingface_hub, "snapshot_download", Mock(return_value="/local/model"))
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=24000))
    monkeypatch.setattr(
        MossTranscribeDiarizeProcessor, "from_pretrained", Mock(return_value=processor)
    )
    prepare = Mock(side_effect=PreparedBoundary())
    monkeypatch.setattr(st, "vocal_waveform", prepare)
    load_model = Mock()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    check = Mock()
    with pytest.raises(PreparedBoundary):
        st.SongTranscriber("model", "revision").transcribe(b"original", check=check)
    prepare.assert_called_once_with(b"original", sample_rate=24000, check=check)
    load_model.assert_not_called()
    assert "overlap2" in st.PIPELINE_REVISION
