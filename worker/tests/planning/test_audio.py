from unittest.mock import MagicMock

from worker.planning.audio import audio_prerequisites, plan_audio

ARTIFACTS = (
    "external_id",
    "track_id",
    "has_muq",
    "has_mert",
    "has_fingerprint",
    "has_melody",
    "has_descriptors",
    "has_waveform",
    "has_visual_features",
)


def test_planner_revision_two_missing_cache_only_requires_shared_mono_decode(monkeypatch):
    from core.config import reset_config_cache

    monkeypatch.delenv("ECHORA_RECORDING_MODEL_MANIFEST", raising=False)
    reset_config_cache()
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    # All older pipelines complete, only the visual cache missing/stale.
    cursor.fetchall.return_value = [
        dict(zip(ARTIFACTS, ("song", "track", True, True, True, True, True, True, False)))
    ]
    plan = plan_audio(connection, "library", ["song"])
    assert plan.visual_feature_external_ids == frozenset({"song"})
    assert not plan.needs_muq and not plan.needs_mert and not plan.needs_melody
    prerequisites = audio_prerequisites(plan, "song")
    assert prerequisites.mono_rates == (22050,)
    assert prerequisites.stereo_rates == ()
    assert not prerequisites.melody
    sql, params = cursor.execute.call_args.args
    assert params[5] == "2"
    assert "vf.revision=%s" in sql
    assert "vf.status IN ('complete', 'unsupported')" in sql
    cursor.fetchall.return_value = [
        dict(zip(ARTIFACTS, ("song", "track", True, True, True, True, True, True, True)))
    ]
    assert not plan_audio(connection, "library", ["song"]).visual_feature_external_ids
