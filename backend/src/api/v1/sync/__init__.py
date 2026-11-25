"""``/sync``: connections, imports, syncs, backfills and library-wide rebuilds."""

from backend.api.endpoint import Category
from backend.api.v1.sync.audio_profiles_rebuild import AudioProfilesRebuild
from backend.api.v1.sync.audio_profiles_status import AudioProfilesStatus
from backend.api.v1.sync.backfill_karaoke import KaraokeBackfill
from backend.api.v1.sync.backfill_lyrics import LyricsBackfill
from backend.api.v1.sync.backfill_recordings import RecordingsBackfill
from backend.api.v1.sync.backfill_voice import VoiceBackfill
from backend.api.v1.sync.catalog import ConnectionCatalog
from backend.api.v1.sync.discover import Discover
from backend.api.v1.sync.import_sample import ImportSample
from backend.api.v1.sync.ingest import Ingest
from backend.api.v1.sync.sample import ConnectionSample
from backend.api.v1.sync.semantic_fusion_rebuild import SemanticFusionRebuild
from backend.api.v1.sync.semantic_fusion_status import SemanticFusionStatus
from backend.api.v1.sync.start import StartSync
from backend.api.v1.sync.status import ConnectionSyncStatus

CATEGORY = Category(
    prefix="/sync",
    endpoints=[
        Discover,
        Ingest,
        ConnectionSample,
        ImportSample,
        ConnectionCatalog,
        ConnectionSyncStatus,
        StartSync,
        RecordingsBackfill,
        LyricsBackfill,
        VoiceBackfill,
        KaraokeBackfill,
        SemanticFusionRebuild,
        SemanticFusionStatus,
        AudioProfilesRebuild,
        AudioProfilesStatus,
    ],
)
