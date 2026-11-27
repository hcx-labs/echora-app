"""``/settings``: the user's settings, the instance's model switches and integrations.

External AI and External processing validation errors never echo the body (``RedactedRoute``).
"""

from backend.api.endpoint import Category
from backend.api.v1.settings._routes import RedactedRoute
from backend.api.v1.settings.external_ai import ExternalAISettingsRead
from backend.api.v1.settings.external_ai_endpoint import ExternalAIEndpoint
from backend.api.v1.settings.external_ai_endpoint_update import ExternalAIEndpointUpdate
from backend.api.v1.settings.external_ai_themes import ExternalAIThemes
from backend.api.v1.settings.external_ai_themes_update import ExternalAIThemesUpdate
from backend.api.v1.settings.external_ai_transcription import ExternalAITranscription
from backend.api.v1.settings.external_ai_transcription_update import ExternalAITranscriptionUpdate
from backend.api.v1.settings.external_ai_translation import ExternalAITranslation
from backend.api.v1.settings.external_ai_translation_update import ExternalAITranslationUpdate
from backend.api.v1.settings.external_ai_translations import ClearTranslations
from backend.api.v1.settings.external_ai_update import ExternalAISettingsUpdate
from backend.api.v1.settings.external_processing import ExternalProcessingRead
from backend.api.v1.settings.external_processing_availability import ModalAvailability
from backend.api.v1.settings.external_processing_prepare import PrepareModal
from backend.api.v1.settings.external_processing_update import ExternalProcessingSettingsUpdate
from backend.api.v1.settings.get import GetSettings
from backend.api.v1.settings.lastfm import UpdateLastFm
from backend.api.v1.settings.lastfm_disconnect import DisconnectLastFm
from backend.api.v1.settings.models_huggingface import HuggingFaceTokenUpdate
from backend.api.v1.settings.models_hum import HumSwitch
from backend.api.v1.settings.models_karaoke import KaraokeSwitch
from backend.api.v1.settings.models_karaoke_stems import KaraokeStemsStatus, KaraokeStemsSwitch
from backend.api.v1.settings.models_transcription import TranscriptionSwitch
from backend.api.v1.settings.motion_artwork import MotionArtworkSettingsRead
from backend.api.v1.settings.motion_artwork_location import MotionArtworkLocation
from backend.api.v1.settings.motion_artwork_loops import ClearMotionArtwork
from backend.api.v1.settings.motion_artwork_status import MotionArtworkStatus
from backend.api.v1.settings.motion_artwork_update import MotionArtworkSettingsUpdate
from backend.api.v1.settings.navidrome import UpdateNavidrome
from backend.api.v1.settings.oidc import OidcSettings
from backend.api.v1.settings.oidc_allow import AllowOidcEmail
from backend.api.v1.settings.oidc_disallow import DisallowOidcEmail
from backend.api.v1.settings.oidc_policy import UpdateOidcPolicy
from backend.api.v1.settings.oidc_user import UpdateOidcUser
from backend.api.v1.settings.onboarding import SaveOnboarding
from backend.api.v1.settings.plugin import PluginSettings
from backend.api.v1.settings.plugin_key_delete import RevokePluginKey
from backend.api.v1.settings.plugin_keys import CreatePluginKey
from backend.api.v1.settings.plugin_update import UpdatePluginSettings
from backend.api.v1.settings.profile import UpdateProfile
from backend.api.v1.settings.runtime import RuntimeSettings
from backend.api.v1.settings.timezone import UpdateTimezone

CATEGORY = Category(
    prefix="/settings",
    route_class=RedactedRoute,
    endpoints=[
        GetSettings,
        RuntimeSettings,
        SaveOnboarding,
        UpdateProfile,
        UpdateTimezone,
        UpdateNavidrome,
        UpdateLastFm,
        DisconnectLastFm,
        KaraokeSwitch,
        KaraokeStemsSwitch,
        KaraokeStemsStatus,
        TranscriptionSwitch,
        HumSwitch,
        HuggingFaceTokenUpdate,
        OidcSettings,
        UpdateOidcPolicy,
        AllowOidcEmail,
        DisallowOidcEmail,
        UpdateOidcUser,
        ExternalAISettingsRead,
        ExternalAISettingsUpdate,
        ExternalAIEndpoint,
        ExternalAIEndpointUpdate,
        ExternalAITranslation,
        ExternalAITranslationUpdate,
        ExternalAITranscription,
        ExternalAITranscriptionUpdate,
        ExternalAIThemes,
        ExternalAIThemesUpdate,
        ClearTranslations,
        ExternalProcessingRead,
        ExternalProcessingSettingsUpdate,
        PrepareModal,
        ModalAvailability,
        MotionArtworkSettingsRead,
        MotionArtworkSettingsUpdate,
        MotionArtworkLocation,
        MotionArtworkStatus,
        ClearMotionArtwork,
        PluginSettings,
        UpdatePluginSettings,
        CreatePluginKey,
        RevokePluginKey,
    ],
)
