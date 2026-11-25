import pytest
from pydantic import SecretStr, ValidationError

from core.config import EchoraConfig, compute, config_classes, reset_config_cache, secret, setting


class _AlignConfig(EchoraConfig, env_prefix="ECHORA_TESTALIGN_"):
    model_id: str = compute("echora/model", legacy="OLD_TESTALIGN_MODEL")
    min_letter_ms: int = compute(40, ge=0)
    timeout_seconds: float = setting(120.0, gt=0, description="How long to wait.")
    token: SecretStr = secret(legacy="OLD_TESTALIGN_TOKEN")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in (
        "ECHORA_TESTALIGN_MODEL_ID",
        "OLD_TESTALIGN_MODEL",
        "ECHORA_TESTALIGN_MIN_LETTER_MS",
        "ECHORA_TESTALIGN_TOKEN",
        "OLD_TESTALIGN_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def test_defaults():
    config = _AlignConfig.load()
    assert (config.model_id, config.min_letter_ms, config.timeout_seconds) == (
        "echora/model",
        40,
        120.0,
    )


def test_every_field_is_its_prefixed_environment_variable(monkeypatch):
    monkeypatch.setenv("ECHORA_TESTALIGN_MIN_LETTER_MS", "55")
    assert _AlignConfig.load().min_letter_ms == 55


def test_a_legacy_name_is_read_after_the_canonical_one(monkeypatch):
    monkeypatch.setenv("OLD_TESTALIGN_MODEL", "legacy/model")
    assert _AlignConfig.load().model_id == "legacy/model"
    monkeypatch.setenv("ECHORA_TESTALIGN_MODEL_ID", "canonical/model")
    reset_config_cache()
    assert _AlignConfig.load().model_id == "canonical/model"


def test_load_is_cached_until_reset(monkeypatch):
    first = _AlignConfig.load()
    monkeypatch.setenv("ECHORA_TESTALIGN_MIN_LETTER_MS", "1")
    assert _AlignConfig.load() is first
    reset_config_cache()
    assert _AlignConfig.load().min_letter_ms == 1


def test_invalid_values_are_rejected_without_echoing_them(monkeypatch):
    monkeypatch.setenv("ECHORA_TESTALIGN_MIN_LETTER_MS", "-12345")
    with pytest.raises(ValidationError) as error:
        _AlignConfig.load()
    assert "-12345" not in str(error.value)


def test_secrets_are_hidden(monkeypatch):
    monkeypatch.setenv("OLD_TESTALIGN_TOKEN", "hf_s3cr3t")
    config = _AlignConfig.load()
    assert config.token.get_secret_value() == "hf_s3cr3t"
    assert "hf_s3cr3t" not in repr(config)
    assert "token" not in config.model_dump()


def test_configs_are_frozen():
    with pytest.raises(ValidationError):
        _AlignConfig.load().min_letter_ms = 1


def test_compute_values_round_trip(monkeypatch):
    monkeypatch.setenv("ECHORA_TESTALIGN_MIN_LETTER_MS", "70")
    values = _AlignConfig.load().compute_values()
    assert values == {"model_id": "echora/model", "min_letter_ms": 70}
    monkeypatch.delenv("ECHORA_TESTALIGN_MIN_LETTER_MS")
    assert _AlignConfig.with_compute_values(values).min_letter_ms == 70


def test_only_compute_fields_are_accepted_remotely():
    with pytest.raises(ValueError, match="timeout_seconds"):
        _AlignConfig.with_compute_values({"timeout_seconds": 1})


def test_catalog_entries():
    entries = {entry.field: entry for entry in _AlignConfig.entries()}
    assert entries["model_id"].name == "ECHORA_TESTALIGN_MODEL_ID"
    assert entries["model_id"].deprecated_names == ("OLD_TESTALIGN_MODEL",)
    assert entries["model_id"].compute
    assert entries["timeout_seconds"].description == "How long to wait."
    assert not entries["timeout_seconds"].compute
    assert entries["token"].secret
    assert entries["token"].default is None
    assert _AlignConfig in config_classes()


def test_a_bare_field_name_is_not_an_environment_variable(monkeypatch):
    class _Model(EchoraConfig, env_prefix="ECHORA_TEST_BARE_"):
        revision: str = setting("pinned", legacy="OLD_TEST_BARE_REVISION")

    monkeypatch.setenv("REVISION", "stray")
    assert _Model().revision == "pinned"
    monkeypatch.setenv("OLD_TEST_BARE_REVISION", "legacy")
    assert _Model().revision == "legacy"
    monkeypatch.setenv("ECHORA_TEST_BARE_REVISION", "canonical")
    assert _Model().revision == "canonical"
    assert _Model(revision="given").revision == "given"


def test_load_all_validates_every_class_of_the_named_modules(monkeypatch):
    from core.config import load_all
    from core.jobs.config import QueueConfig

    reset_config_cache()
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/echora")
    load_all(("core.db.config", "core.jobs.config"))
    assert QueueConfig in EchoraConfig._instances

    reset_config_cache()
    field = next(e for e in QueueConfig.entries() if e.field == "lease_seconds")
    monkeypatch.setenv(field.name, "not a number")
    with pytest.raises(ValidationError):
        load_all(("core.jobs.config",))
    reset_config_cache()
