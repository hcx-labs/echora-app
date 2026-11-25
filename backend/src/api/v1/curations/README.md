# Curations (`/api/v1/curations`)

Saved curations, which Echora keeps as Navidrome playlists, and previews of recipes and
journeys. A curation's playlist is built by its refresh job (`worker/src/curation/`); writes to
a saved curation take `curation_jobs.locked` so they never race a running refresh.

| Endpoint | File | |
| --- | --- | --- |
| `GET /curations` | `list.py` | the user's curations with their latest revision's songs |
| `POST /curations` | `create.py` | save a recipe and queue its first refresh (201) |
| `PUT /curations/{id}` | `replace.py` | replace the recipe and queue a new revision (202) |
| `PATCH /curations/{id}` | `update.py` | schedule, refresh mode, size, familiarity |
| `DELETE /curations/{id}` | `delete.py` | delete, and its Navidrome playlist unless `delete_navidrome=false` (204) |
| `POST /curations/{id}/refresh` | `refresh.py` | queue a new revision now (202) |
| `POST /curations/preview` | `preview.py` | the songs a recipe selects now (shuffled per call) |
| `POST /curations/journeys/preview` | `journey_preview.py` | a path of songs between two songs |

The recipe model is `core.curation.preview.CurationRecipe`; request bodies reject unknown fields.
