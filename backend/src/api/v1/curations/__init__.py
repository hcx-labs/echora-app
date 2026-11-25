"""``/curations``: saved curations (Navidrome playlists), recipe previews and journeys."""

from backend.api.endpoint import Category
from backend.api.v1.curations.create import CreateCuration
from backend.api.v1.curations.delete import DeleteCuration
from backend.api.v1.curations.journey_preview import JourneyPreview
from backend.api.v1.curations.list import ListCurations
from backend.api.v1.curations.preview import PreviewCuration
from backend.api.v1.curations.refresh import RefreshCuration
from backend.api.v1.curations.replace import ReplaceCuration
from backend.api.v1.curations.update import UpdateCuration

CATEGORY = Category(
    prefix="/curations",
    endpoints=[
        ListCurations,
        CreateCuration,
        PreviewCuration,
        JourneyPreview,
        ReplaceCuration,
        UpdateCuration,
        DeleteCuration,
        RefreshCuration,
    ],
)
