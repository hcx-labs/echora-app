# Plugin (`/api/v1/plugin`)

The provider API of the Echora Navidrome plugin (`navidrome/`). Responses are Navidrome's
provider envelopes. The plugin's base URL is `…/api/v1/plugin`.

| Endpoint | File | |
| --- | --- | --- |
| `POST /plugin/similar-tracks` | `similar_tracks.py` | `{matches: [{song, similarity}]}` |
| `POST /plugin/similar-artists` | `similar_artists.py` | `{artists: [{id, name}]}` |
| `POST /plugin/artist-radio` | `artist_radio.py` | `{songs: [...]}` |
| `POST /plugin/sonic-path` | `sonic_path.py` | `{matches}`, similarity -1 per step |
| `POST /plugin/lyrics` | `lyrics.py` | `{lyrics: [{lang, text}]}`: TTML or LRC, empty to fall back |

**Auth.** `Authorization: Bearer echora_nd_…`, an integration key issued in Settings
(`/api/v1/settings/integrations/navidrome/keys`). Only its SHA-256 digest is stored. A key works
while it is unrevoked and unexpired, its user is not blocked and the user's integration is enabled
for the key's Navidrome connection; otherwise 401 with `WWW-Authenticate: Bearer`
(`backend/src/settings/plugin_integration.py`). A key sees only its user's songs in its
connection's library (`backend/src/plugin/scope.py`).

**Requests** are the PDK's request structs, which carry metadata Echora does not use: unknown
fields are ignored at every level (`_models.py`), while known fields are still validated.

**Ranking** (`backend/src/search/plugin.py`) uses existing embeddings only. The corpus is a
persistent search service per user and library, reloaded when a visible song, its metadata,
recording group or current embedding changes. Ranking weights, the missing-lyrics policy, the
per-artist cap and the lyrics format come from the integration's ranking profile.
