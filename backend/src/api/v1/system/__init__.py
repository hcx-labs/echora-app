"""``/system``: health and capabilities."""

from backend.api.endpoint import Category
from backend.api.v1.system.capabilities import Capabilities
from backend.api.v1.system.health import Health

CATEGORY = Category(prefix="/system", endpoints=[Health, Capabilities])
