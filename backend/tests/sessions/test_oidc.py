"""Account provisioning on OIDC sign-in."""

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from core.config import reset_config_cache
from core.db.models import OidcAllowedEmail, OidcSetting, User


@pytest.fixture
def oidc(accounts, monkeypatch):
    monkeypatch.setenv("ECHORA_OIDC_BOOTSTRAP_ADMIN_EMAIL", "Admin@Example.test")
    reset_config_cache()
    import backend.sessions.oidc as oidc

    return oidc


def claims(email, sub=None, **extra):
    return {"sub": sub or "sub-" + email, "email": email, **extra}


def refusal(oidc, value):
    with pytest.raises(HTTPException) as error:
        oidc.sign_in(value)
    return error.value.status_code, error.value.detail


def test_bootstrap_administrator_signs_in_first(oidc, accounts):
    scope, _ = accounts
    assert refusal(oidc, claims("someone@example.test")) == (
        403,
        "The bootstrap administrator must sign in first",
    )
    token, _ = oidc.sign_in(claims("ADMIN@example.test", name="Admin"))
    assert token
    with scope() as session:
        admin = session.scalar(select(User).where(User.email == "admin@example.test"))
        assert admin.is_admin and admin.display_name == "Admin"
        assert admin.preference is not None


def test_policy_and_allow_list_gate_new_accounts(oidc, accounts):
    scope, _ = accounts
    oidc.sign_in(claims("admin@example.test"))
    # Without a policy row, only approved emails get accounts (the settings page shows the
    # policy as on until it is first saved).
    assert refusal(oidc, claims("unlisted@example.test"))[0] == 403
    with scope() as session:
        session.add(OidcSetting(singleton=True, auto_provision=True))
    oidc.sign_in(claims("open@example.test"))
    with scope() as session:
        session.get(OidcSetting, True).auto_provision = False
    assert refusal(oidc, claims("closed@example.test"))[0] == 403
    with scope() as session:
        session.add(OidcAllowedEmail(email="closed@example.test"))
    oidc.sign_in(claims("closed@example.test"))
    with scope() as session:
        assert session.get(OidcAllowedEmail, "closed@example.test") is None
        user = session.scalar(select(User).where(User.email == "closed@example.test"))
        assert not user.is_admin


def test_existing_accounts_keep_their_identity(oidc, accounts):
    scope, _ = accounts
    oidc.sign_in(claims("admin@example.test", sub="a"))
    assert refusal(oidc, claims("admin@example.test", sub="b")) == (
        409,
        "This email is linked to another OIDC identity",
    )
    with scope() as session:
        session.scalar(select(User).where(User.email == "admin@example.test")).is_blocked = True
    assert refusal(oidc, claims("admin@example.test", sub="a"))[0] == 403


def test_claims_are_checked(oidc, monkeypatch):
    assert refusal(oidc, {"sub": "x"})[0] == 422
    monkeypatch.setenv("ECHORA_OIDC_REQUIRE_VERIFIED_EMAIL", "true")
    reset_config_cache()
    assert refusal(oidc, claims("admin@example.test")) == (403, "OIDC email is not verified")


def test_client_only_with_complete_configuration(no_oidc, monkeypatch):
    import backend.sessions.oidc as oidc

    assert oidc.oidc_client() is None
    for name, value in [
        ("ECHORA_OIDC_ISSUER_URL", "https://id.example.test/"),
        ("ECHORA_OIDC_CLIENT_ID", "echora"),
        ("ECHORA_OIDC_CLIENT_SECRET", "secret"),
    ]:
        monkeypatch.setenv(name, value)
    reset_config_cache()
    assert oidc.oidc_client() is not None
    assert oidc.issuer() == "https://id.example.test"
