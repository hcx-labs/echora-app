"""``/library``: the user's songs and what Echora knows about them, artists, the map, concepts."""

from backend.api.endpoint import Category
from backend.api.v1.library.artist_profile import ArtistProfile
from backend.api.v1.library.artists_similar import ArtistsSimilar
from backend.api.v1.library.community_snapshot import CommunitySnapshot
from backend.api.v1.library.community_snapshots import CommunitySnapshots
from backend.api.v1.library.concept_create import ConceptCreateEndpoint
from backend.api.v1.library.concept_preview import ConceptPreviewEndpoint
from backend.api.v1.library.concepts import Concepts
from backend.api.v1.library.facets import FacetsEndpoint
from backend.api.v1.library.filter_options import FilterOptionsEndpoint
from backend.api.v1.library.lyrics_status import LyricsStatus
from backend.api.v1.library.map import LibraryMap
from backend.api.v1.library.top_tracks import TopTracksEndpoint
from backend.api.v1.library.track_audio_descriptors import TrackAudioDescriptors
from backend.api.v1.library.track_audio_profile import TrackAudioProfile
from backend.api.v1.library.track_audio_quality import TrackAudioQuality
from backend.api.v1.library.track_info import TrackInfoEndpoint
from backend.api.v1.library.track_lyrics import TrackLyricsEndpoint
from backend.api.v1.library.track_lyrics_status import TrackLyricsStatus
from backend.api.v1.library.track_lyrics_update import TrackLyricsUpdate
from backend.api.v1.library.track_motion_artwork import TrackMotionArtwork
from backend.api.v1.library.track_recording_group import TrackRecordingGroup
from backend.api.v1.library.track_similar import TrackSimilar
from backend.api.v1.library.track_transcription_language import TrackTranscriptionLanguage
from backend.api.v1.library.track_visual_features import TrackVisualFeatures
from backend.api.v1.library.track_waveform import TrackWaveform
from backend.api.v1.library.tracks import LibraryTracks

CATEGORY = Category(
    prefix="/library",
    endpoints=[
        LibraryTracks,
        TrackInfoEndpoint,
        TrackSimilar,
        TrackRecordingGroup,
        TrackAudioDescriptors,
        TrackAudioQuality,
        TrackAudioProfile,
        TrackWaveform,
        TrackVisualFeatures,
        TrackMotionArtwork,
        TrackLyricsEndpoint,
        TrackLyricsUpdate,
        TrackLyricsStatus,
        TrackTranscriptionLanguage,
        LyricsStatus,
        ArtistProfile,
        ArtistsSimilar,
        LibraryMap,
        CommunitySnapshots,
        CommunitySnapshot,
        FilterOptionsEndpoint,
        FacetsEndpoint,
        TopTracksEndpoint,
        Concepts,
        ConceptCreateEndpoint,
        ConceptPreviewEndpoint,
    ],
)
