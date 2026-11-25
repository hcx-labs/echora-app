# auth (`/api/v1/auth`)

Sign-in through OpenID Connect, sign-out, the current user and the session's deadlines.

| Endpoint | File | Needs |
| --- | --- | --- |
| `GET /oidc/status` | `oidc_status.py` | nothing |
| `GET /oidc/start` | `oidc_start.py` | nothing (503 without OIDC) |
| `GET /oidc/callback` | `oidc_callback.py` | the provider's redirect; sets `echora_session` |
| `POST /logout` | `logout.py` | nothing; clears the cookie |
| `GET /me` | `me.py` | a session |
| `GET /session` | `session.py` | a session; never renews it |
| `POST /session/activity` | `session_activity.py` | a session, `X-Echora-Activity: 1`, a trusted `Origin` |

The callback's URL is the OIDC redirect URI: `ECHORA_OIDC_REDIRECT_URI` and the provider's
registered redirect must be `https://<host>/api/v1/auth/oidc/callback`. Account provisioning and
the session store are in [`backend.sessions`](../../../sessions/README.md).
