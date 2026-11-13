from echora_analysis.transcription_evidence import retry_evidence
from echora_analysis.transcription_recovery import recover_window, WindowDecodeError


def activity(score=0.01):
    return {
        "method": "discogs-effnet_voice_instrumental",
        "duration_seconds": 60,
        "windows": [
            {"start_seconds": i, "end_seconds": i + 2, "vocal_activation": score}
            for i in range(0, 60, 2)
        ],
    }


def test_requires_both_quiet_stem_and_low_activity():
    assert retry_evidence(activity(), 0, 60, 60, -60)
    assert retry_evidence(activity(0.6), 0, 60, 60, -60) is None
    assert retry_evidence(activity(), 0, 60, 60, -20) is None


def test_missing_stale_or_sparse_evidence_does_not_suppress_retries():
    assert retry_evidence(None, 0, 60, 60, -60) is None
    assert retry_evidence(activity(), 0, 60, 100, -60) is None
    a = activity()
    a["windows"] = a["windows"][:5]
    assert retry_evidence(a, 0, 60, 60, -60) is None


def test_single_vocal_patch_preserves_retries():
    a = activity()
    a["windows"][2]["vocal_activation"] = 0.8
    assert retry_evidence(a, 0, 60, 60, -60) is None


def test_first_attempt_always_runs_before_evidence():
    calls = []
    gaps = []

    def decode(a, b):
        calls.append((a, b))
        raise WindowDecodeError("loop")

    assert (
        recover_window(
            decode,
            0,
            60,
            on_unresolved=gaps.append,
            retry_policy=lambda a, b: {"reason": "low_vocal_evidence"},
        )
        == []
    )
    assert calls == [(0, 60)]
    assert gaps[0]["retry_evidence"]["reason"] == "low_vocal_evidence"


def test_success_does_not_consult_evidence():
    def policy(a, b):
        raise AssertionError("Should not run")

    assert recover_window(lambda a, b: ["lyrics"], 0, 60, retry_policy=policy) == ["lyrics"]


def _activity(values):
    return {
        "method": "discogs-effnet_voice_instrumental",
        "windows": [
            {"start_seconds": i * 2.0, "end_seconds": i * 2.0 + 2.064, "vocal_activation": v}
            for i, v in enumerate(values)
        ],
    }


def test_instrumental_song_needs_few_vocal_windows_and_low_mean():
    from echora_analysis.transcription_evidence import instrumental_song

    quiet = [0.02] * 95 + [0.8] * 2 + [0.3] * 3
    assert instrumental_song(_activity(quiet))["decision"] == "instrumental"
    assert instrumental_song(_activity([0.02] * 90 + [0.9] * 10)) is None  # 10% clearly vocal
    assert instrumental_song(_activity([0.2] * 100)) is None  # mean too high
    assert instrumental_song(_activity([0.01] * 10)) is None  # too little evidence
    assert instrumental_song(_activity([0.01] * 50 + [float("nan")])) is None
    assert instrumental_song({"method": "other", "windows": []}) is None
    assert instrumental_song(None) is None
