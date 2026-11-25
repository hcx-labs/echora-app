"""``/auth``: OIDC sign-in, sign-out, the current user and the session's deadlines."""

from backend.api.endpoint import Category
from backend.api.v1.auth.logout import Logout
from backend.api.v1.auth.me import CurrentUser
from backend.api.v1.auth.oidc_callback import OidcCallback
from backend.api.v1.auth.oidc_start import OidcStart
from backend.api.v1.auth.oidc_status import OidcStatus
from backend.api.v1.auth.session import ReadSession
from backend.api.v1.auth.session_activity import RecordActivity

CATEGORY = Category(
    prefix="/auth",
    endpoints=[
        OidcStatus,
        OidcStart,
        OidcCallback,
        Logout,
        CurrentUser,
        ReadSession,
        RecordActivity,
    ],
)
