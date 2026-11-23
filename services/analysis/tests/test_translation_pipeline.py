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


def test_declined_counted_apart_from_failed(monkeypatch):
    import echora_analysis
    from contextlib import contextmanager
    from echora_analysis.lyric_translation import TranslationError

    settings = ExternalAISettings(
        enabled=True,
        url="http://localhost/v1",
        model="test",
        language_pairs=[{"source": "ja", "target": "en"}, {"source": "ja", "target": "fr"}],
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
    monkeypatch.setattr(pipeline, "load_translations", lambda *a: [])
    errors = iter([TranslationError("declined"), TranslationError("transport_failure")])

    def translate(*a, **kw):
        raise next(errors)

    monkeypatch.setattr(pipeline, "translate", translate)
    result = pipeline.backfill_translations(
        "user", "url", ["id"], check=lambda: None, progress=lambda _: None
    )
    assert (result["declined"], result["failed"]) == (1, 1)


def test_mixed_song_translates_only_its_foreign_lines(monkeypatch):
    import echora_analysis
    from contextlib import contextmanager

    settings = ExternalAISettings(
        enabled=True,
        url="http://localhost/v1",
        model="test",
        language_pairs=[{"source": "ur", "target": "en"}, {"source": "es", "target": "en"}],
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: (settings, None))
    monkeypatch.setattr(
        echora_analysis, "main", SimpleNamespace(_cipher=lambda: None), raising=False
    )
    lines = ["When I'm with you", "Mera bhi usse batana", "Eenie, meenie", "Tere bina"]
    row = {"track_id": "track", "text": "\n".join(lines), "language": "en", "provenance": {}}
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row]
    db.execute.return_value.fetchone.return_value = {"text": row["text"], "provenance": {}}

    @contextmanager
    def connect(*a, **kw):
        yield db

    monkeypatch.setattr(pipeline.psycopg, "connect", connect)
    monkeypatch.setattr(pipeline, "require_database_url", lambda: "unused")
    monkeypatch.setattr(pipeline, "load_translations", lambda *a: [])
    # The rule flags the Urdu lines and the vocable; the model leaves the vocable as it is.
    monkeypatch.setattr(pipeline, "foreign_lines", lambda lines, language, sources: [1, 2, 3])
    answer = [
        {"id": 0, "text": "When I'm with you"},
        {"id": 1, "text": "Tell him about me too"},
        {"id": 2, "text": "Eenie, meenie!"},
        {"id": 3, "text": "Without you"},
    ]
    calls = []

    def translate(settings, key, pair, lines, mixed=False):
        calls.append((pair, mixed))
        return answer

    saved = []
    monkeypatch.setattr(pipeline, "translate", translate)
    monkeypatch.setattr(pipeline, "save_translation", lambda *a, **kw: saved.append(a))
    result = pipeline.backfill_translations(
        "user", "url", ["id"], check=lambda: None, progress=lambda _: None
    )
    assert result["translated"] == 1
    assert [(p.source, p.target, mixed) for p, mixed in calls] == [("mul", "en", True)]
    assert [line["text"] for line in saved[0][3]] == [
        "",
        "Tell him about me too",
        "",
        "Without you",
    ]


def test_mixed_needs_two_foreign_lines(monkeypatch):
    import echora_analysis
    from contextlib import contextmanager

    settings = ExternalAISettings(
        enabled=True,
        url="http://localhost/v1",
        model="test",
        language_pairs=[{"source": "ur", "target": "en"}],
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: (settings, None))
    monkeypatch.setattr(
        echora_analysis, "main", SimpleNamespace(_cipher=lambda: None), raising=False
    )
    row = {"track_id": "track", "text": "a\nb", "language": "en", "provenance": {}}
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [row]

    @contextmanager
    def connect(*a, **kw):
        yield db

    monkeypatch.setattr(pipeline.psycopg, "connect", connect)
    monkeypatch.setattr(pipeline, "require_database_url", lambda: "unused")
    monkeypatch.setattr(pipeline, "foreign_lines", lambda *a: [1])
    translate = MagicMock()
    monkeypatch.setattr(pipeline, "translate", translate)
    pipeline.backfill_translations(
        "user", "url", ["id"], check=lambda: None, progress=lambda _: None
    )
    translate.assert_not_called()


def test_foreign_lines_uses_the_rule_and_confident_lid(monkeypatch):
    rules = SimpleNamespace(
        _LID_WORD=__import__("re").compile(r"[^\W\d_]+(?:'[^\W\d_]+)*"),
        line_language=lambda line, language: "auto" if "bhi" in line else language,
        line_lid=lambda text: {
            "y me cansé de luchar": ("es", 0.79),
            "my broken pieces": ("pt", 0.85),
        }.get(text, ("en", 0.9)),
    )
    monkeypatch.setattr(pipeline, "_label_rules", lambda: rules)
    lines = ["Mera bhi usse", "Y me cansé de luchar", "My broken pieces", "Hello there you", ""]
    # Portuguese is not translated from, so that guess is ignored.
    assert pipeline.foreign_lines(lines, "en", {"es", "ur"}) == [0, 1]
