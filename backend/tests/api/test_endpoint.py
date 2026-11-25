import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from backend.api.endpoint import Category, Endpoint
from backend.api.requests import PatchRequest, PostRequest
from backend.api.responses import ResponseModel


class Body(PostRequest):
    name: str


class Echo(ResponseModel):
    name: str
    note: str | None = None


class EchoEndpoint(Endpoint):
    """Echo the name."""

    method = "POST"
    path = "/echo"
    request = Body
    response = Echo
    status_code = 201

    def handle(self, body: Body) -> dict:
        return {"name": body.name}


def _client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(Category(prefix="/t", endpoints=[EchoEndpoint]).router())
    return TestClient(app)


def test_endpoint_routes_validates_and_leaves_unset_fields_out():
    response = _client().post("/t/echo", json={"name": " a "})
    assert response.status_code == 201
    assert response.json() == {"name": " a "}


def test_request_rejects_unknown_fields():
    assert _client().post("/t/echo", json={"name": "a", "other": 1}).status_code == 422


def test_summary_comes_from_the_docstring():
    router = APIRouter()
    EchoEndpoint.register(router)
    assert router.routes[0].summary == "Echo the name."


def test_handle_must_take_the_declared_body():
    with pytest.raises(TypeError):

        class Wrong(Endpoint):
            method = "POST"
            path = "/x"
            request = Body

            def handle(self) -> None: ...


def test_patch_changes_only_the_sent_fields():
    class Update(PatchRequest):
        a: int | None = None
        b: int | None = None

    assert Update(a=1).changes() == {"a": 1}


def test_insecure_cookies_refused_off_localhost(monkeypatch):
    from backend.app import enforce_secure_cookie_policy

    monkeypatch.setenv(
        "ECHORA_OIDC_REDIRECT_URI", "https://music.example.test/api/v1/auth/oidc/callback"
    )
    with pytest.raises(RuntimeError):
        enforce_secure_cookie_policy()
    monkeypatch.setenv("ECHORA_SESSION_COOKIE_SECURE", "true")
    from core.config import reset_config_cache

    reset_config_cache()
    enforce_secure_cookie_policy()
