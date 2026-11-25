import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

import backend.api.v1.settings.external_ai_endpoint_update as endpoint_update
import backend.api.v1.settings.external_ai_update as settings_update
import core.lyrics.translations as storage
from backend.api.v1.settings.external_ai_endpoint_update import EndpointUpdate
from backend.api.v1.settings.external_ai_update import ExternalAIUpdate
from core.integrations.external_ai import ExternalAISettings

PATH = "/api/v1/settings/external-ai"


def config(**values):
    return ExternalAISettings(
        enabled=True,
        url="http://localhost:8000/v1",
        model="local",
        language_pairs=[{"source": "JA", "target": "en_US"}],
        **values,
    )


@pytest.fixture
def state(monkeypatch):
    state = {"settings": ExternalAISettings(), "encrypted": None}
    cipher = Fernet(Fernet.generate_key())

    def save(settings, encrypted, *, replace_key):
        state["settings"] = settings
        if replace_key:
            state["encrypted"] = encrypted
        return bool(state["encrypted"])

    monkeypatch.setattr(storage, "load_settings", lambda: (state["settings"], state["encrypted"]))
    monkeypatch.setattr(storage, "save_settings", save)
    monkeypatch.setattr(settings_update, "credential_cipher", lambda: cipher)
    monkeypatch.setattr(endpoint_update, "credential_cipher", lambda: cipher)
    state["cipher"] = cipher
    return state


SECTIONS = ["", "/endpoint", "/translation", "/transcription", "/themes"]


@pytest.mark.parametrize("section", SECTIONS)
def test_sessions_only(anonymous, state, section):
    assert anonymous.get(PATH + section).status_code == 401
    assert anonymous.put(PATH + section, json={}).status_code == 401


@pytest.mark.parametrize("section", SECTIONS)
def test_administrators_only(client, state, section):
    assert client.get(PATH + section).status_code == 403
    assert client.put(PATH + section, json={}).status_code == 403


def test_auth_and_encryption_redaction(client, admin, state):
    assert client.get(PATH).json()["enabled"] is False
    result = client.put(PATH, json={**config().model_dump(), "api_key": "private-marker"})
    assert result.status_code == 200
    assert result.json()["has_key"]
    assert "private-marker" not in result.text + client.get(PATH).text
    assert state["cipher"].decrypt(state["encrypted"]) == b"private-marker"
    client.put(PATH, json=config().model_dump())
    assert state["encrypted"]
    assert client.put(PATH, json={"url": "file:///tmp/x"}).status_code == 422
    client.put(PATH, json={"api_key": ""})
    assert not client.get(PATH).json()["has_key"]


def test_invalid_body_does_not_echo_key(client, admin, state):
    for body in [
        {"enabled": True, "api_key": "private-marker"},
        {"api_key": "private-marker", "unknown": "private-marker"},
        {"api_key": "private-marker" * 1000},
    ]:
        result = client.put(PATH, json=body)
        assert result.status_code == 422
        assert result.json() == {"detail": "Invalid External AI settings"}


@pytest.mark.parametrize(
    "section,keys",
    [
        ("endpoint", {"enabled", "provider", "url", "model", "has_key"}),
        ("translation", {"prompt", "language_pairs"}),
        ("themes", {"enabled", "lyric_themes"}),
        (
            "transcription",
            {
                "enabled",
                "transcription_refinement",
                "transcription_prompt",
                "transcription_model",
                "model",
                "default_transcription_prompt",
            },
        ),
    ],
)
def test_section_contracts(client, admin, state, monkeypatch, section, keys):
    path = f"{PATH}/{section}"
    assert set(client.get(path).json()) == keys
    combined = {**ExternalAISettings().model_dump(), "has_key": True}
    monkeypatch.setattr(storage, "save_section", lambda *a, **kw: combined)
    assert set(client.put(path, json={}).json()) == keys
    wrong = {"prompt": "wrong-section"} if section == "endpoint" else {"url": "http://localhost/v1"}
    assert client.put(path, json=wrong).status_code == 422


def test_endpoint_key_is_encrypted_and_replaced_only_when_sent(client, admin, state, monkeypatch):
    calls = []

    def save_section(values, encrypted=None, *, replace_key=False):
        calls.append((values, encrypted, replace_key))
        return {**ExternalAISettings().model_dump(), "has_key": True}

    monkeypatch.setattr(storage, "save_section", save_section)
    client.put(f"{PATH}/endpoint", json={"provider": "anthropic", "api_key": "k"})
    client.put(f"{PATH}/endpoint", json={"provider": "anthropic"})
    (values, encrypted, replace), (_, none, keep) = calls
    assert "api_key" not in values and replace and not keep and none is None
    assert state["cipher"].decrypt(encrypted) == b"k"


def test_clear_translations_requires_admin_and_idle_jobs(client, user, monkeypatch):
    calls = []
    monkeypatch.setattr(storage, "clear_translations", lambda: calls.append(True) or 7)
    path = f"{PATH}/translations"
    assert client.delete(path).status_code == 403
    assert calls == []
    user["is_admin"] = True
    assert client.delete(path).json() == {"deleted": 7}

    def busy():
        raise RuntimeError("translation_jobs_active")

    monkeypatch.setattr(storage, "clear_translations", busy)
    assert client.delete(path).status_code == 409


def test_key_updates_never_dump_the_key():
    assert ExternalAIUpdate(api_key="secret").model_dump().get("api_key") is None


def test_endpoint_update_needs_a_url_only_for_openai():
    with pytest.raises(ValidationError):
        EndpointUpdate(enabled=True, provider="openai")
    assert EndpointUpdate(enabled=True, provider="anthropic").provider == "anthropic"
