"""Fixtures for the core tests."""

import pytest

from core.config import reset_config_cache
from core.testing import database  # noqa: F401 - shared fixture


@pytest.fixture(autouse=True)
def reset_settings_cache():
    reset_config_cache()
    yield
    reset_config_cache()
