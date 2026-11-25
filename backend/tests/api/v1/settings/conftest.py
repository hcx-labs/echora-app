import pytest


@pytest.fixture
def admin(user):
    """Makes the signed-in user an administrator."""
    user["is_admin"] = True
    return user
