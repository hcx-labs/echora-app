import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest

import worker.media.preprocessing as p
from core.config import reset_config_cache


def cache(tmp_path, **kwargs):
    return p.AudioArtifactCache(tmp_path, **kwargs)


def test_cache_reused_across_instances_and_not_mutable(tmp_path):
    produce = Mock(return_value=np.ones((10, 2), dtype=np.float32))
    first = cache(tmp_path).array(b"track", {"kind": "vocals", "revision": "one"}, produce)
    first[:] = 9
    actual = cache(tmp_path).array(b"track", {"kind": "vocals", "revision": "one"}, produce)
    assert np.all(actual == 1)
    assert produce.call_count == 1


def test_source_and_recipe_invalidate(tmp_path):
    produce = Mock(return_value=np.ones(10, dtype=np.float32))
    c = cache(tmp_path)
    for source, recipe in [
        (b"one", {"rate": 16000}),
        (b"two", {"rate": 16000}),
        (b"two", {"rate": 24000}),
    ]:
        c.array(source, recipe, produce)
    assert produce.call_count == 3


@pytest.mark.parametrize("damage", ["array", "manifest", "manifest_list", "missing_manifest"])
def test_corruption_is_rebuilt(tmp_path, damage):
    produce = Mock(return_value=np.ones(10, dtype=np.float32))
    c = cache(tmp_path)
    c.array(b"song", {}, produce)
    if damage == "array":
        next(tmp_path.glob("*.npy")).write_bytes(b"truncated")
    elif damage == "manifest":
        next(tmp_path.glob("*.json")).write_text("{bad json")
    elif damage == "manifest_list":
        next(tmp_path.glob("*.json")).write_text("[]")
    else:
        next(tmp_path.glob("*.json")).unlink()
    np.testing.assert_array_equal(c.array(b"song", {}, produce), np.ones(10))
    assert produce.call_count == 2


def test_cancel_during_publish_never_reuses_partial(tmp_path):
    class Cancel(BaseException):
        pass

    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if checks == 5:  # After array publish, before manifest publish.
            raise Cancel()

    produce = Mock(return_value=np.ones(10, dtype=np.float32))
    with pytest.raises(Cancel):
        cache(tmp_path, check=check).array(b"song", {}, produce)
    assert not list(tmp_path.glob(".*.tmp"))
    cache(tmp_path).array(b"song", {}, produce)
    assert produce.call_count == 2


def test_failed_or_nonfinite_producer_is_not_cached(tmp_path):
    with pytest.raises(ValueError):
        cache(tmp_path).array(b"song", {}, lambda: np.array([np.nan]))
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(RuntimeError):
        cache(tmp_path).array(b"song", {}, Mock(side_effect=RuntimeError("broken")))
    assert not list(tmp_path.glob("*.npy"))


def test_concurrent_same_artifact_has_one_producer(tmp_path):
    barrier = threading.Barrier(2)
    produce = Mock(return_value=np.ones(1000, dtype=np.float32))

    def read():
        barrier.wait(timeout=5)
        return cache(tmp_path).array(b"same", {}, produce)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read) for _ in range(2)]
        for future in futures:
            np.testing.assert_array_equal(future.result(timeout=5), np.ones(1000))
    assert produce.call_count == 1


def test_ttl_and_lru_budget(tmp_path):
    c = cache(tmp_path, ttl_seconds=10, max_bytes=20000)
    c.array(b"old", {}, lambda: np.ones(1000, dtype=np.float32))
    old = next(tmp_path.glob("*.npy"))
    os.utime(old, (1, 1))
    c.prune()
    assert not old.exists()
    c.array(b"one", {}, lambda: np.ones(1000, dtype=np.float32))
    c.array(b"two", {}, lambda: np.ones(1000, dtype=np.float32))
    c.max_bytes = 5000
    c.prune()
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_oversized_result_is_returned_without_retention(tmp_path):
    c = cache(tmp_path, max_bytes=0)
    assert c.array(b"song", {}, lambda: np.ones(1000)).shape == (1000,)
    assert not list(tmp_path.glob("*.npy"))


def test_decode_contracts_cached_only_in_session(monkeypatch, tmp_path):
    import worker.media.audio as audio

    monkeypatch.setenv("ECHORA_PREPROCESS_DIR", str(tmp_path))
    reset_config_cache()
    raw = Mock(return_value=np.ones(10, dtype=np.float32))
    stereo = Mock(return_value=np.ones((10, 2), dtype=np.float32))
    monkeypatch.setattr(audio, "_decode_audio", raw)
    monkeypatch.setattr(audio, "_decode_audio_channels", stereo)
    with p.preprocessing_session():
        for _ in range(2):
            audio.decode_audio(b"song", 24000)
            audio.decode_audio_channels(b"song", 24000, 1)
            audio.decode_audio_channels(b"song", 24000, 2)
        assert raw.call_count == stereo.call_count == 1
        audio.decode_audio(b"song", 16000)
        assert raw.call_count == 2
    assert p.active_cache() is None
    audio.decode_audio(b"song", 24000)
    assert raw.call_count == 3


def test_shared_vocals_for_transcription(monkeypatch, tmp_path):
    import worker.media.audio as audio
    import worker.media.roformer as roformer

    monkeypatch.setenv("ECHORA_PREPROCESS_DIR", str(tmp_path))
    reset_config_cache()
    separate = Mock(side_effect=lambda waveform, check: waveform * 0.25)
    monkeypatch.setattr(roformer, "separate_vocals", separate)
    monkeypatch.setattr(
        audio, "_decode_audio_channels", Mock(return_value=np.ones((44100, 2), dtype=np.float32))
    )
    before_separate = Mock()
    with p.preprocessing_session():
        p.vocal_waveform(b"song", before_separate=before_separate)
        p.prepare_audio(b"song", vocals=True)
        transcript = p.vocal_waveform(b"song", 16000, before_separate=before_separate)
    before_separate.assert_called_once()
    assert separate.call_count == 1
    with p.preprocessing_session():
        np.testing.assert_array_equal(p.vocal_waveform(b"song", 16000), transcript)
    assert separate.call_count == 1
    monkeypatch.setattr(roformer, "SEPARATION_REVISION", "new-settings")
    with p.preprocessing_session():
        p.vocal_waveform(b"song", 16000)
    assert separate.call_count == 2


def test_cancel_check_propagates_through_session(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHORA_PREPROCESS_DIR", str(tmp_path))
    reset_config_cache()

    class Cancel(BaseException):
        pass

    with p.preprocessing_session(lambda: (_ for _ in ()).throw(Cancel())):
        with pytest.raises(Cancel):
            p.cached_decode(b"song", 24000, 1, lambda: np.ones(10))
    assert p.active_cache() is None


def test_manifest_has_source_and_recipe_but_no_credentials(tmp_path):
    cache(tmp_path).array(b"private-audio", {"kind": "decode", "channels": 1}, lambda: np.ones(10))
    metadata = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert metadata["source_sha256"]
    assert metadata["recipe"]["channels"] == 1
    assert "private-audio" not in json.dumps(metadata)


def test_audio_plan_prepares_only_pending_formats():
    from worker.planning.audio import AudioProcessingPlan, audio_prerequisites

    plan = AudioProcessingPlan(
        frozenset({"embedding"}),
        frozenset({"embedding"}),
        frozenset({"fingerprint"}),
        frozenset({"melody"}),
        frozenset({"melody"}),
        frozenset({"waveform"}),
    )
    assert audio_prerequisites(plan, "embedding").mono_rates == (24000,)
    assert audio_prerequisites(plan, "melody").stereo_rates == (44100,)
    assert audio_prerequisites(plan, "waveform").stereo_rates == (24000,)
    assert audio_prerequisites(plan, "fingerprint").mono_rates == ()
    assert audio_prerequisites(plan, "current").stereo_rates == ()


def test_melody_reuses_shared_vocals_and_caches_residual(monkeypatch, tmp_path):
    import worker.media.audio as audio
    import worker.media.roformer as roformer

    monkeypatch.setenv("ECHORA_PREPROCESS_DIR", str(tmp_path))
    reset_config_cache()
    mix = np.random.default_rng(22).uniform(-2, 2, (44100, 2)).astype(np.float32)
    decode = Mock(return_value=mix.copy())
    separate = Mock(side_effect=lambda waveform, check: waveform * 0.25)
    monkeypatch.setattr(audio, "_decode_audio_channels", decode)
    monkeypatch.setattr(roformer, "separate_vocals", separate)
    with p.preprocessing_session():
        p.prepare_audio(b"song", melody=True)
        vocals, accompaniment = p.melody_waveforms(b"song")
        p.vocal_waveform(b"song", 16000)  # Transcription reuses the same separation.
    with p.preprocessing_session():
        again, residual = p.melody_waveforms(b"song")
    assert separate.call_count == decode.call_count == 1
    np.testing.assert_array_equal(vocals, again)
    np.testing.assert_array_equal(accompaniment, residual)
    np.testing.assert_allclose(vocals + accompaniment, mix.mean(axis=1), atol=2e-7)
    assert accompaniment.max() > 1  # Preserve float range, without clipping/normalization.
    manifests = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert any(m["recipe"]["kind"] == "accompaniment-mono" for m in manifests)
    monkeypatch.setattr(roformer, "SEPARATION_REVISION", "different-recipe")
    with p.preprocessing_session():
        p.melody_waveforms(b"song")
    assert separate.call_count == 2
    assert decode.call_count == 1


def test_prune_stops_checking_once_remaining_entries_are_retained(tmp_path):
    check = Mock()
    c = cache(tmp_path, check=check)
    for index in range(100):
        np.save(tmp_path / f"{index:064x}.npy", np.ones(10, dtype=np.float32))
    c.prune()
    assert len(list(tmp_path.glob("*.npy"))) == 100
    # The callback can perform a database heartbeat; its cost must not grow
    # with the number of artifacts when no eviction is necessary.
    assert check.call_count <= 3


@pytest.mark.parametrize("frames", [44099, 44100, 44101, 100003])
def test_karaoke_reference_matches_vocal_resampling_grid(monkeypatch, tmp_path, frames):
    import worker.media.audio as audio
    import worker.media.roformer as roformer

    monkeypatch.setenv("ECHORA_PREPROCESS_DIR", str(tmp_path))
    reset_config_cache()
    mix = np.random.default_rng(frames).uniform(-1, 1, (frames, 2)).astype(np.float32)
    decode = Mock(return_value=mix)
    separate = Mock(side_effect=lambda waveform, check: waveform.copy())
    monkeypatch.setattr(audio, "_decode_audio_channels", decode)
    monkeypatch.setattr(roformer, "separate_vocals", separate)
    with p.preprocessing_session():
        p.prepare_audio(b"song", vocals=True, reference=True)
        vocals = p.vocal_waveform(b"song")
        reference = p.vocal_reference_waveform(b"song")
    np.testing.assert_array_equal(reference, vocals)
    assert decode.call_count == separate.call_count == 1
    with p.preprocessing_session():
        np.testing.assert_array_equal(p.vocal_reference_waveform(b"song"), reference)
    assert decode.call_count == 1
