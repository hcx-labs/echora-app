from unittest.mock import Mock

import backend.api.v1.auth.logout as logout


def test_me_returns_only_public_fields(client, user):
    assert client.get("/api/v1/auth/me").json() == {
        "username": user["username"],
        "email": user["email"],
        "display_name": user["display_name"],
        "is_admin": False,
        "onboarding_complete": True,
        "navidrome_connection_id": None,
    }


def test_me_needs_a_session(anonymous):
    assert anonymous.get("/api/v1/auth/me").status_code == 401


def test_logout_deletes_the_session_and_clears_the_cookie(anonymous, monkeypatch):
    deleted = Mock()
    monkeypatch.setattr(logout, "delete_session", deleted)
    anonymous.cookies.set("echora_session", "token")
    response = anonymous.post("/api/v1/auth/logout")
    assert response.status_code == 204
    deleted.assert_called_once_with("token")
    cookie = response.headers["set-cookie"]
    assert cookie.startswith('echora_session=""') and "Path=/" in cookie


def test_logout_without_a_session(anonymous, monkeypatch):
    deleted = Mock()
    monkeypatch.setattr(logout, "delete_session", deleted)
    assert anonymous.post("/api/v1/auth/logout").status_code == 204
    deleted.assert_not_called()


def test_oidc_unconfigured(anonymous):
    assert anonymous.get("/api/v1/auth/oidc/status").json() == {
        "configured": False,
        "provider": "OIDC",
    }
    assert anonymous.get("/api/v1/auth/oidc/start").status_code == 503
    assert anonymous.get("/api/v1/auth/oidc/callback").status_code == 503


def test_callback_route_keeps_its_name(app):
    assert app.url_path_for("oidc_callback") == "/api/v1/auth/oidc/callback"
