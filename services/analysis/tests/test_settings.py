"""The settings catalog across every config class: names, defaults, validation and secrecy."""

import json
import os
import runpy
from pathlib import Path

import pytest
from core.config import reset_config_cache
from core.db.config import DatabaseConfig, require_database_url
from core.errors import ConfigurationError
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
EXPORTER = runpy.run_path(str(ROOT / "scripts/export-runtime-settings.py"))
CLASSES = {cls.__name__: cls for cls in EXPORTER["catalog"]()}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    names = {
        name
        for cls in CLASSES.values()
        for entry in cls.entries()
        for name in (entry.name, *entry.deprecated_names)
    }
    for name in os.environ:
        if name.upper() in names:
            monkeypatch.delenv(name)
    reset_config_cache()
    yield
    reset_config_cache()


def test_every_variable_has_one_owner():
    seen: dict[str, str] = {}
    for cls in CLASSES.values():
        for entry in cls.entries():
            for name in (entry.name, *entry.deprecated_names):
                assert name not in seen, f"{name}: {seen.get(name)} and {cls.__name__}"
                seen[name] = cls.__name__


def test_defaults_and_no_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DATABASE_URL=do-not-read\nECHORA_BATCH_SIZE=1\n")
    monkeypatch.chdir(tmp_path)
    assert DatabaseConfig().database_url == ""
    assert CLASSES["WorkerConfig"]().batch_size == 128
    assert CLASSES["QueueConfig"]().lease_seconds == 120
    assert CLASSES["OidcConfig"]().redirect_uri == ""


def test_database_url_is_required_by_connection_entry_points(monkeypatch):
    with pytest.raises(ConfigurationError, match="DATABASE_URL is required"):
        require_database_url()
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/echora")
    reset_config_cache()
    assert require_database_url() == "postgresql://example.invalid/echora"


def test_legacy_names_case_and_cache(monkeypatch):
    worker, session = CLASSES["WorkerConfig"], CLASSES["SessionConfig"]
    monkeypatch.setenv("echora_batch_size", "12")
    monkeypatch.setenv("COOKIE_SECURE", "true")
    assert worker.load().batch_size == 12
    assert session.load().cookie_secure is True
    monkeypatch.setenv("echora_batch_size", "24")
    assert worker.load().batch_size == 12
    reset_config_cache()
    assert worker.load().batch_size == 24
    # The canonical name wins over a legacy one.
    monkeypatch.setenv("ECHORA_WORKER_BATCH_SIZE", "36")
    reset_config_cache()
    assert worker.load().batch_size == 36


def test_secrets_do_not_export(monkeypatch):
    for name in (
        "DATABASE_URL",
        "REDIS_URL",
        "OIDC_SESSION_SECRET",
        "OIDC_CLIENT_SECRET",
        "CREDENTIAL_ENCRYPTION_KEY",
    ):
        monkeypatch.setenv(name, "private-marker")
    for cls in CLASSES.values():
        config = cls()
        assert "private-marker" not in repr(config)
        assert "private-marker" not in config.model_dump_json()
    assert DatabaseConfig().database_url == "private-marker"
    oidc = CLASSES["OidcConfig"]()
    assert oidc.client_secret.get_secret_value() == "private-marker"
    assert "private-marker" not in EXPORTER["template"]()


def test_generated_session_secret_is_stable_per_process():
    oidc = CLASSES["OidcConfig"]
    assert oidc().session_secret.get_secret_value() == oidc().session_secret.get_secret_value()


@pytest.mark.parametrize(
    ("owner", "values"),
    [
        ("WorkerConfig", {"batch_size": 0}),
        ("DatabaseConfig", {"db_pool_size": 0}),
        ("DatabaseConfig", {"db_max_overflow": -1}),
        ("QueueConfig", {"lease_seconds": 14}),
        ("QueueConfig", {"retry_max_seconds": 1}),
        ("SessionConfig", {"activity_check_seconds": 3600}),
        ("SessionConfig", {"renew_interval_seconds": 604800}),
        ("SessionConfig", {"absolute_timeout_seconds": 604799}),
        ("SessionConfig", {"activity_window_seconds": 59}),
        ("SessionConfig", {"request_timeout_seconds": 0}),
        ("SessionConfig", {"trusted_origins": "*"}),
        ("ApiConfig", {"cors_origins": "*"}),
        ("FusionConfig", {"weight_lyrics": 0, "weight_audio": 0}),
        ("FusionConfig", {"weight_audio": float("nan")}),
        ("KaraConfig", {"audio_speed": 0}),
        ("KaraConfig", {"audio_speed": 1.51}),
        ("KaraConfig", {"aligner": "unknown"}),
        ("KaraConfig", {"job_timeout_seconds": 0}),
        ("PreprocessConfig", {"max_bytes": -1}),
        ("PreprocessConfig", {"ttl_seconds": 0}),
        ("RecordingConfig", {"max_upload_bytes": 0}),
        ("NavidromeConfig", {"max_connections": 2}),
        ("MediaCacheConfig", {"timeout_seconds": 0}),
        ("UiConfig", {"recording_poll_ms": 0}),
    ],
)
def test_invalid_settings(owner, values):
    with pytest.raises(ValidationError):
        CLASSES[owner](**values)


def test_errors_hide_input(monkeypatch):
    monkeypatch.setenv("ECHORA_BATCH_SIZE", "private-invalid-value")
    with pytest.raises(ValidationError) as error:
        CLASSES["WorkerConfig"]()
    assert "private-invalid-value" not in str(error.value)


def test_empty_models_preserved(monkeypatch):
    for name in (
        "MUQ_MODEL_ID",
        "MERT_MODEL_ID",
        "LYRICS_MODEL_ID",
        "MOSS_MODEL_ID",
        "FA_KARA_MODEL_ID",
        "ECHORA_RECORDING_MODEL_MANIFEST",
    ):
        monkeypatch.setenv(name, "")
    embedding = CLASSES["EmbeddingConfig"]()
    assert embedding.muq_model_id == embedding.mert_model_id == embedding.lyrics_model_id == ""
    assert CLASSES["TranscriptionConfig"]().moss_model_id == ""
    assert CLASSES["KaraConfig"]().model_id == ""
    assert CLASSES["RecordingConfig"]().model_manifest == ""
    assert CLASSES["NavidromeConfig"](audio_cache_bytes=0)
    assert CLASSES["PreprocessConfig"](max_bytes=0)
    assert CLASSES["LibraryConfig"](source_recheck_seconds=0)


def test_public_ui_is_only_the_ui_config(monkeypatch):
    ui = CLASSES["UiConfig"]
    expected = {
        "job_poll_ms": 1200,
        "batch_poll_ms": 1500,
        "hum_index_poll_ms": 2500,
        "recording_poll_ms": 1500,
        "recording_search_timeout_ms": 120000,
        "recording_request_timeout_ms": 10000,
        "recording_upload_timeout_ms": 60000,
    }
    assert ui.load().public_ui() == expected
    monkeypatch.setenv("ECHORA_JOB_POLL_MS", "42")
    assert ui().public_ui() == {**expected, "job_poll_ms": 42}


def test_operational_http_settings(monkeypatch):
    from core.integrations.navidrome import NavidromeClient

    monkeypatch.setenv("ECHORA_NAVIDROME_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("ECHORA_NAVIDROME_READ_TIMEOUT_SECONDS", "9")
    reset_config_cache()
    with NavidromeClient("https://example.invalid", "user", "pass") as client:
        assert client.client.timeout.connect == 7
        assert client.client.timeout.read == 9


def test_environment_catalog_matches_settings_defaults():
    assert (ROOT / "configs/runtime.env.example").read_text() == EXPORTER["template"]()


def test_browser_defaults_match_settings():
    defaults = ROOT / "web/components/runtime/runtimeConfig.defaults.json"
    assert json.loads(defaults.read_text()) == EXPORTER["ui_defaults"]()
