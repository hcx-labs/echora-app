import pytest
from pydantic import ValidationError

from core.settings import Settings, get_settings, require_database_url


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    import os

    names = {name.upper() for name in Settings.model_fields}
    names.update(field.validation_alias for field in Settings.model_fields.values()
                 if isinstance(field.validation_alias, str))
    for name in os.environ:
        if name.upper() in names:
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_and_no_dotenv(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('DATABASE_URL=do-not-read\nECHORA_BATCH_SIZE=1\n')
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.database_url == ''
    assert settings.batch_size == 128
    assert settings.oidc_redirect_uri == ''
    assert settings.worker_lease_seconds == 120
    assert settings.model_config['env_file'] is None
    assert Settings(unknown='ignored').batch_size == 128


def test_database_url_is_required_by_connection_entry_points(monkeypatch):
    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        require_database_url()
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/echora")
    get_settings.cache_clear()
    assert require_database_url() == "postgresql://example.invalid/echora"


def test_aliases_types_case_and_cache(monkeypatch):
    monkeypatch.setenv('echora_batch_size', '12')
    monkeypatch.setenv('COOKIE_SECURE', 'true')
    assert get_settings().batch_size == 12
    assert get_settings().cookie_secure is True
    monkeypatch.setenv('echora_batch_size', '24')
    assert get_settings().batch_size == 12
    get_settings.cache_clear()
    assert get_settings().batch_size == 24
    assert Settings(batch_size=3).batch_size == 3


def test_secret_strings_do_not_export():
    names = ('database_url', 'redis_url', 'oidc_session_secret', 'oidc_client_secret',
             'credential_encryption_key')
    settings = Settings(**{name: 'private-marker' for name in names})
    assert settings.database_url == 'private-marker'
    assert 'private-marker' not in repr(settings)
    assert 'private-marker' not in settings.model_dump_json()
    assert all(name not in settings.model_dump() for name in names)
    assert Settings().oidc_session_secret == Settings().oidc_session_secret


@pytest.mark.parametrize('values', [
    {'batch_size': 0}, {'db_pool_size': 0}, {'db_max_overflow': -1},
    {'worker_lease_seconds': 14}, {'worker_heartbeat_seconds': 60},
    {'session_activity_check_seconds': 3600}, {'session_renew_interval_seconds': 604800},
    {'session_absolute_timeout_seconds': 604799}, {'session_activity_window_seconds': 59},
    {'session_request_timeout_seconds': 0}, {'job_retry_max_seconds': 1},
    {'semantic_fusion_weight_lyrics': 0, 'semantic_fusion_weight_audio': 0},
    {'semantic_fusion_weight_audio': float('nan')}, {'fa_kara_audio_speed': 0},
    {'preprocess_max_bytes': -1}, {'preprocess_ttl_seconds': 0},
    {'fa_kara_audio_speed': 1.51}, {'fa_kara_aligner': 'unknown'},
    {'recording_max_upload_bytes': 0}, {'cors_origins': '*'},
    {'session_trusted_origins': '*'}, {'karaoke_job_timeout_seconds': 0},
    {'navidrome_max_connections': 2}, {'media_cache_timeout_seconds': 0},
])
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        Settings(**values)


def test_errors_hide_input(monkeypatch):
    monkeypatch.setenv('ECHORA_BATCH_SIZE', 'private-invalid-value')
    with pytest.raises(ValidationError) as error:
        Settings()
    assert 'private-invalid-value' not in str(error.value)


def test_empty_models_preserved(monkeypatch):
    for name in ('MUQ_MODEL_ID', 'MERT_MODEL_ID', 'LYRICS_MODEL_ID', 'MOSS_MODEL_ID',
                 'FA_KARA_MODEL_ID', 'ECHORA_RECORDING_MODEL_MANIFEST'):
        monkeypatch.setenv(name, '')
    settings = Settings()
    for attr in ('muq_model_id', 'mert_model_id', 'lyrics_model_id', 'moss_model_id',
                 'fa_kara_model_id', 'recording_model_manifest'):
        assert getattr(settings, attr) == ''
    assert Settings(audio_cache_bytes=0, preprocess_max_bytes=0, source_recheck_seconds=0)


def test_public_ui_is_an_explicit_allowlist():
    from core.settings import public_ui_settings

    expected = {
        "job_poll_ms": 1200, "batch_poll_ms": 1500, "hum_index_poll_ms": 2500,
        "recording_poll_ms": 1500, "recording_search_timeout_ms": 120000,
        "recording_request_timeout_ms": 10000, "recording_upload_timeout_ms": 60000,
    }
    assert public_ui_settings() == expected
    assert Settings(database_url="private", job_poll_ms=42).public_ui() == {
        **expected, "job_poll_ms": 42,
    }
    with pytest.raises(ValidationError):
        Settings(recording_poll_ms=0)


def test_operational_http_settings(monkeypatch):
    from core.integrations.navidrome import NavidromeClient

    monkeypatch.setenv("ECHORA_NAVIDROME_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("ECHORA_NAVIDROME_READ_TIMEOUT_SECONDS", "9")
    get_settings.cache_clear()
    with NavidromeClient("https://example.invalid", "user", "pass") as client:
        assert client.client.timeout.connect == 7
        assert client.client.timeout.read == 9


def test_environment_catalog_matches_settings_defaults():
    import runpy
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    exporter = runpy.run_path(str(root / 'scripts/export-runtime-settings.py'))
    assert (root / 'configs/runtime.env.example').read_text() == exporter['template']()


def test_browser_defaults_match_settings():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    defaults = root / 'web/components/runtime/runtimeConfig.defaults.json'
    assert json.loads(defaults.read_text()) == Settings.model_construct().public_ui()
