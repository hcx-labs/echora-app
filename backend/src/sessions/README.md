# Sessions (`backend.sessions`)

App sessions and OpenID Connect sign-in.

- `store.py`: sessions in `user_sessions`, stored by the SHA-256 of their token. A session has an
  idle deadline (`ECHORA_SESSION_IDLE_TIMEOUT_SECONDS`) and an absolute one. Reads never move the
  idle deadline; only explicit activity (`POST /api/v1/auth/session/activity`) renews it, at most
  once per renew interval and never past the absolute deadline. `session_user` resolves the
  `echora_session` cookie for `backend.api.deps.current_user`; blocked users and users without
  preferences have no session (401).
- `oidc.py`: the provider client (`oidc_client`, `None` when `ECHORA_OIDC_*` is incomplete) and
  `sign_in`, which finds or provisions the account for the provider's claims and opens a session.

Configuration: `SessionConfig` and `OidcConfig` in `backend/src/config.py`.
