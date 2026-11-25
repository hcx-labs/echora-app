"""``/media``: streams, cover art, motion artwork loops and scrobbles."""

from backend.api.endpoint import Category
from backend.api.v1.media.cover import Cover
from backend.api.v1.media.motion_artwork import MotionArtworkVideo
from backend.api.v1.media.scrobble import Scrobble
from backend.api.v1.media.stream import Stream

CATEGORY = Category(prefix="/media", endpoints=[Stream, Cover, MotionArtworkVideo, Scrobble])
