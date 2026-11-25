import pytest

from core.config import reset_config_cache

OIDC_SETTINGS = (
    "ISSUER_URL",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "REDIRECT_URI",
    "POST_LOGIN_REDIRECT",
    "BOOTSTRAP_ADMIN_EMAIL",
    "REQUIRE_VERIFIED_EMAIL",
)


@pytest.fixture(autouse=True)
def no_oidc(monkeypatch):
    """OIDC unconfigured, whatever the environment running the tests sets."""
    for name in OIDC_SETTINGS:
        monkeypatch.delenv(f"ECHORA_OIDC_{name}", raising=False)
        monkeypatch.delenv(f"OIDC_{name}", raising=False)
    reset_config_cache()
