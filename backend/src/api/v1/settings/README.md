# settings (`/api/v1/settings`)

The user's own settings, and the instance's settings that administrators manage.

| Endpoints | Files | Needs |
| --- | --- | --- |
| `GET ""` (everything the settings page shows), `PUT /onboarding`, `/profile`, `/timezone`, `/navidrome`, `PUT`/`DELETE /lastfm` | `get.py`, `onboarding.py`, `profile.py`, `timezone.py`, `navidrome.py`, `lastfm*.py` | a session |
| `GET /runtime` (browser polling intervals) | `runtime.py` | nothing |
| `PUT /models/{karaoke,karaoke-stems,transcription,hum,huggingface}`, `GET /models/karaoke-stems` (stored stems) | `models_*.py` | an administrator |
| `GET /oidc`, `PUT /oidc/policy`, `POST`/`DELETE /oidc/allowed-emails`, `PATCH /oidc/users/{id}` | `oidc*.py` | an administrator |
| `GET`/`PUT /external-ai` and its sections `endpoint`, `translation`, `transcription`, `themes`; `DELETE /external-ai/translations` | `external_ai*.py` | an administrator |
| `GET`/`PUT /external-processing`, `POST /external-processing/prepare` | `external_processing*.py` | an administrator |
| `GET /external-processing/availability` | `external_processing_availability.py` | a session |
| `GET`/`PUT /motion-artwork`, `PUT /motion-artwork/location`, `GET /motion-artwork/status`, `DELETE /motion-artwork/loops` | `motion_artwork*.py` | an administrator |
| `GET`/`PUT /integrations/navidrome`, `POST /integrations/navidrome/keys`, `DELETE /integrations/navidrome/keys/{id}` | `plugin*.py` | a session |

Secrets (API keys, tokens, passwords) are stored encrypted and never returned; responses say
only whether one is stored. Validation errors of the External AI and External processing routes
are replaced by a fixed message (`_routes.py`), since Pydantic's can echo the body.

Two administrator checks exist with different messages: `backend.api.deps.admin_user` for OIDC
("Administrator access is required") and `_deps.admin_required` for the rest ("Administrator
access required").

The plugin's ranking profile, key issuing and key authentication are in
`backend/src/settings/plugin_integration.py`.
