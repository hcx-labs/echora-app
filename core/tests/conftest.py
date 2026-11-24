"""Fixtures for the core tests."""

import pytest

from core.settings import get_settings
from core.testing import database  # noqa: F401 - shared fixture


@pytest.fixture(autouse=True)
def reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
