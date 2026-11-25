"""``/plugin``: the Navidrome plugin's provider API (bearer integration keys)."""

from backend.api.endpoint import Category
from backend.api.v1.plugin.artist_radio import ArtistRadio
from backend.api.v1.plugin.lyrics import Lyrics
from backend.api.v1.plugin.similar_artists import SimilarArtists
from backend.api.v1.plugin.similar_tracks import SimilarTracks
from backend.api.v1.plugin.sonic_path import SonicPath

CATEGORY = Category(
    prefix="/plugin",
    endpoints=[SimilarTracks, SimilarArtists, ArtistRadio, SonicPath, Lyrics],
    tags=["navidrome-plugin"],
)
