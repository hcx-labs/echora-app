from unittest.mock import MagicMock
from types import SimpleNamespace
import pytest
from echora_analysis.external_ai import ExternalAISettings
from echora_analysis import translation_pipeline as pipeline


def test_line_selection():
    assert pipeline.source_lines({"text": "a\nb"}) == ["a", "b"]
    assert pipeline.source_lines(
        {"text": "fallback", "provenance": {"lines": [{"text": "timed"}]}}
    ) == ["timed"]


def test_disabled_never_calls_endpoint(monkeypatch):
    monkeypatch.setattr(pipeline, "load_settings", lambda: (ExternalAISettings(), None))
    translate = MagicMock()
    monkeypatch.setattr(pipeline, "translate", translate)
    result = pipeline.backfill_translations(
        "user", "url", [], check=lambda: None, progress=lambda _: None
    )
    assert result["translated"] == 0
    translate.assert_not_called()


def test_cancellation_and_reuse(monkeypatch):
    import echora_analysis
    from contextlib import contextmanager

    settings = ExternalAISettings(
        enabled=True,
        url="http://localhost/v1",
        model="test",
        language_pairs=[{"source": "ja", "target": "en"}],
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: (settings, None))
    monkeypatch.setattr(
        echora_analysis, "main", SimpleNamespace(_cipher=lambda: None), raising=False
    )
    row = {"track_id": "track", "text": "source", "language": "ja", "provenance": {}}
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row]

    @contextmanager
    def connect(*a, **kw):
        yield db

    monkeypatch.setattr(pipeline.psycopg, "connect", connect)
    monkeypatch.setattr(pipeline, "require_database_url", lambda: "unused")
    monkeypatch.setattr(
        pipeline,
        "load_translations",
        lambda *a: [{"source_language": "ja", "target_language": "en", "status": "ready"}],
    )
    translate = MagicMock()
    monkeypatch.setattr(pipeline, "translate", translate)
    result = pipeline.backfill_translations(
        "user", "url", ["id"], check=lambda: None, progress=lambda _: None
    )
    assert result["reused"] == 1
    translate.assert_not_called()

    class Cancelled(BaseException):
        pass

    def cancel():
        raise Cancelled()

    with pytest.raises(Cancelled):
        pipeline.backfill_translations("user", "url", ["id"], check=cancel, progress=lambda _: None)
