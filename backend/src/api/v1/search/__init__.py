"""``/search``: hum, recording and concept search."""

from backend.api.endpoint import Category
from backend.api.v1.search.hum import HumSearchEndpoint
from backend.api.v1.search.hum_index_build import HumIndexBuild
from backend.api.v1.search.hum_index_status import HumIndexStatus

CATEGORY = Category(prefix="/search", endpoints=[HumSearchEndpoint, HumIndexStatus, HumIndexBuild])
