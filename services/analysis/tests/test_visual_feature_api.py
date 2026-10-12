"""Real endpoint body with fake IO, independent of heavyweight app/model imports.

Database-backed coverage also lives in test_representations.py.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import uuid

import pytest

from echora_analysis.settings import Settings
from echora_analysis.melody_preview import melody_preview
from echora_analysis.processing_plan import audio_prerequisites, plan_audio
from echora_analysis.visual_features import VISUAL_FEATURE_REVISION


class HttpError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code


def endpoint(rows):
    source = Path(__file__).parents[1] / "src/echora_analysis/main.py"
    tree = ast.parse(source.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "track_visual_features"
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    connect = MagicMock()
    cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = rows
    session = MagicMock(return_value={"id": "user"})
    namespace = dict(
        get_settings=lambda: Settings.model_construct(database_url="fake"),
        Cookie=lambda **kw: None,
        _session_user=session,
        psycopg=SimpleNamespace(connect=connect),
        os=SimpleNamespace(environ={"DATABASE_URL": "fake"}),
        dict_row=object(),
        HTTPException=HttpError,
        VISUAL_FEATURE_REVISION=VISUAL_FEATURE_REVISION,
        DESCRIPTOR_REVISION="1",
        MELODY_CONTOUR_REVISION="melody-current",
        melody_preview=melody_preview,
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["track_visual_features"], cursor, session, connect


def test_api_authentication_precedes_database_access():
    api, cursor, session, connect = endpoint([])
    session.side_effect = HttpError(401, "login")
    with pytest.raises(HttpError) as error:
        api(uuid.uuid4(), None)
    assert error.value.status_code == 401
    connect.assert_not_called()


def test_api_visibility_precedes_feature_and_enrichment_reads():
    api, cursor, session, _ = endpoint([None])
    track = uuid.uuid4()
    with pytest.raises(HttpError) as error:
        api(track, "cookie")
    assert error.value.status_code == 404
    session.assert_called_once_with("cookie")
    cursor.execute.assert_called_once()
    assert cursor.execute.call_args.args[1] == ("user", track)


def test_api_returns_optional_current_enrichment_without_writes_or_model_work():
    descriptor = {"revision": "1", "status": "complete", "descriptors": {"rhythm": {"bpm": 120}}}
    vocal = {"activity": {"windows": []}, "run_id": "voice-run"}
    contour = {"source": "full-mix", "pitch": [69, 69], "voiced": [True, True], "hop_seconds": 0.01}
    api, cursor, _, _ = endpoint(
        [
            {"exists": 1},
            {"status": "complete", "features": {"revision": "2"}},
            descriptor,
            vocal,
            contour,
        ]
    )
    track = uuid.uuid4()
    result = api(track, "cookie")
    assert result["status"] == "complete"
    assert result["enrichment"]["descriptors"] == descriptor
    assert result["enrichment"]["vocal_activity"] == vocal
    assert result["enrichment"]["melody"]["points"][0]["pitch"] == 69
    calls = cursor.execute.call_args_list
    assert all(call.args[0].lstrip().startswith("SELECT") for call in calls)
    assert calls[1].args[1] == (track, "2")
    assert "current_analysis_runs" in calls[3].args[0]
    assert calls[4].args[1] == (track, "melody-current")


def test_api_pending_cache_does_not_hide_existing_enrichment():
    api, _, _, _ = endpoint([{"exists": 1}, None, {"status": "complete"}, None, None])
    result = api(uuid.uuid4(), "cookie")
    assert result["status"] == "pending"
    assert result["visual_features"] is None
    assert result["enrichment"]["descriptors"]["status"] == "complete"


def test_api_reports_unsupported_cache_without_serving_empty_features():
    cache = {"status": "unsupported", "features": {}}
    api, _, _, _ = endpoint([{"exists": 1}, cache, None, None, None])
    result = api(uuid.uuid4(), "cookie")
    assert result["status"] == "unsupported"
    assert result["visual_features"] is None


def test_planner_revision_two_missing_cache_only_requires_shared_mono_decode(monkeypatch):
    from echora_analysis.settings import get_settings

    monkeypatch.delenv("ECHORA_RECORDING_MODEL_MANIFEST", raising=False)
    get_settings.cache_clear()
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    # All older pipelines complete, only the visual cache missing/stale.
    cursor.fetchall.return_value = [("song", "track", True, True, True, True, True, True, False)]
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
    cursor.fetchall.return_value = [("song", "track", True, True, True, True, True, True, True)]
    assert not plan_audio(connection, "library", ["song"]).visual_feature_external_ids
