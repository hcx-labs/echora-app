import pytest

import backend.api.v1.settings.external_processing_availability as availability
from backend.api.v1.settings.external_processing_update import ExternalProcessingUpdate


def test_settings_reject_malformed_tokens_and_gpus():
    with pytest.raises(ValueError):
        ExternalProcessingUpdate(token_id="as-not-an-id")
    with pytest.raises(ValueError):
        ExternalProcessingUpdate(gpu="RTX9000")


def test_invalid_body_does_not_echo_the_secret(client, admin):
    response = client.put(
        "/api/v1/settings/external-processing",
        json={"token_id": "bad", "token_secret": "private-marker"},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid External processing settings"}


def test_settings_are_for_administrators(client):
    assert client.get("/api/v1/settings/external-processing").status_code == 403
    assert client.post("/api/v1/settings/external-processing/prepare").status_code == 403


def test_availability_for_every_user(client, monkeypatch):
    monkeypatch.setattr(availability, "compute_for", lambda user, requested: "local")
    assert client.get("/api/v1/settings/external-processing/availability").json() == {
        "available": False,
        "default": "local",
    }

    def refuse(user, requested):
        raise PermissionError

    monkeypatch.setattr(availability, "compute_for", refuse)
    assert client.get("/api/v1/settings/external-processing/availability").json() == {
        "available": False,
        "default": "local",
    }
