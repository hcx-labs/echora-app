import numpy as np

from worker.profiles.pipeline import (
    AudioProfileParameters,
    WindowVector,
    derive_audio_profile,
    infer_window_ranges,
)


def _windows(vectors: list[list[float]]) -> list[WindowVector]:
    return [
        WindowVector(index, index * 5.0, index * 5.0 + 10.0, np.asarray(vector, dtype=np.float32))
        for index, vector in enumerate(vectors)
    ]


def test_consistent_track_stays_one_segment_and_one_mode() -> None:
    profile = derive_audio_profile(_windows([[1.0, 0.0]] * 8), 45.0)

    assert len(profile.segments) == 1
    assert len(profile.modes) == 1
    assert np.isclose(profile.resultant_length, 1.0)
    assert np.isclose(profile.modes[0].duration_weight, 1.0)


def test_distinct_halves_form_segments_and_recurring_modes() -> None:
    parameters = AudioProfileParameters(
        minimum_segment_seconds=15.0,
        minimum_mode_seconds=10.0,
        minimum_mode_share=0.1,
    )
    profile = derive_audio_profile(
        _windows([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4),
        45.0,
        parameters=parameters,
    )

    assert len(profile.segments) == 2
    assert len(profile.modes) == 2
    assert np.isclose(sum(mode.duration_weight for mode in profile.modes), 1.0)
    assert profile.segments[0].end_seconds == profile.segments[1].start_seconds


def test_repeated_identity_produces_disjoint_mode_intervals() -> None:
    parameters = AudioProfileParameters(
        minimum_segment_seconds=10.0,
        minimum_mode_seconds=10.0,
        minimum_mode_share=0.1,
    )
    profile = derive_audio_profile(
        _windows([[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2 + [[1.0, 0.0]] * 2),
        35.0,
        parameters=parameters,
    )

    assert len(profile.modes) == 2
    repeated = max(profile.modes, key=lambda mode: mode.duration_weight)
    assert len(repeated.intervals) == 2


def test_legacy_range_reconstruction_preserves_exact_final_window() -> None:
    assert infer_window_ranges(4, 23.0, 10.0, 5.0) == [
        (0.0, 10.0),
        (5.0, 15.0),
        (10.0, 20.0),
        (13.0, 23.0),
    ]


def test_store_profile_accepts_dictionary_rows_for_profile_and_mode_ids():
    import uuid
    from unittest.mock import MagicMock
    from worker.profiles.pipeline import _store_profile

    profile = derive_audio_profile(_windows([[1.0, 0.0]] * 8), 45.0)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    profile_id, mode_id = uuid.uuid4(), uuid.uuid4()
    cursor.fetchone.side_effect = [{"id": profile_id}, {"id": mode_id}]
    _store_profile(connection, uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), "muq_mulan", profile)
    cursor.execute.assert_any_call("DELETE FROM audio_modes WHERE profile_id=%s", (profile_id,))
    assert cursor.executemany.call_args.args[1][0][0] == mode_id


def test_profile_processor_summary_and_run(monkeypatch):
    import uuid
    from unittest.mock import MagicMock

    import worker.profiles.pipeline as profiles
    from worker.processing import ProcessingContext, run_stage

    good, empty = uuid.uuid4(), uuid.uuid4()
    run, source = uuid.uuid4(), uuid.uuid4()
    planning = MagicMock()
    planning.__enter__.return_value = planning
    planning.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        {
            "track_id": track,
            "source_run_id": source,
            "duration_seconds": 20.0,
            "window_seconds": None,
            "stride_seconds": None,
        }
        for track in (good, empty)
    ]
    working = MagicMock()
    windows = working.cursor.return_value.__enter__.return_value
    window = {
        "window_index": 0,
        "window_start_seconds": 0.0,
        "window_end_seconds": 10.0,
        "embedding": "[1,0]",
    }
    windows.fetchall.side_effect = [[window], []]
    connections = iter([planning, working])
    monkeypatch.setattr(profiles.psycopg, "connect", lambda *a, **kw: next(connections))
    monkeypatch.setattr(profiles, "_create_profile_run", lambda *a: run)
    monkeypatch.setattr(profiles, "derive_audio_profile", lambda *a: "profile")
    stored = []
    monkeypatch.setattr(profiles, "_store_profile", lambda conn, *args: stored.append(args))
    monkeypatch.setattr(
        "core.embeddings.representations.configure_representations", lambda connection: None
    )
    summary = run_stage(
        profiles.AudioProfileProcessor([good, empty], "mert"), ProcessingContext()
    ).as_dict()
    # The summary keys a profile build always had.
    assert summary == {
        "total": 2,
        "profiled": 1,
        "failed": 1,
        "model": "mert",
        "profile_run_id": str(run),
    }
    assert stored == [(good, source, run, "mert", "profile")]
    completed = [c for c in windows.execute.call_args_list if "status='complete'" in c.args[0]]
    assert completed and completed[0].args[1] == (run,)
