# Sync (`/api/v1/sync`)

Connecting Navidrome, analyzing a library and rebuilding library-wide data. Every endpoint needs
a session; a connection id must belong to the signed-in user (404 otherwise). The POSTs queue a
job and return its snapshot (`202`); an equivalent active job is returned instead of a new one
(`existing: true`). Job payloads never carry credentials.

| Endpoint | File | What it does |
| --- | --- | --- |
| `POST /discover` | `discover.py` | Check a server answers, save the connection, preview random songs |
| `POST /ingest` | `ingest.py` | Save a connection and analyze chosen songs |
| `GET /connections/{id}/sample` | `sample.py` | Random songs of a saved connection (onboarding) |
| `POST /connections/{id}/import` | `import_sample.py` | Analyze chosen songs of a saved connection |
| `GET /connections/{id}/catalog` | `catalog.py` | Navidrome's albums and artists |
| `GET /connections/{id}/status` | `status.py` | Analyzed and missing songs, recording fingerprint coverage |
| `POST /connections/{id}` | `start.py` | Sync the connection (`mode`: `all` or `missing`; `compute`) |
| `POST /connections/{id}/backfills/{recordings,lyrics,voice,karaoke}` | `backfill_*.py` | Run one operation's stages over the connection |
| `GET`, `POST /semantic-fusion` | `semantic_fusion_*.py` | Fused vector counts; rebuild them |
| `GET`, `POST /audio-profiles` | `audio_profiles_*.py` | Audio profile coverage; rebuild them |

The four backfills have no caller in the web app. They are kept as operator tools until that is
decided (see the design doc's dead code section).

`GET /connections/{id}/status` scans the whole Navidrome library, and heals renumbered sources
(`relink_renumbered_sources`) on the way, so it is not free of writes.

Helpers live in `backend/src/sync/connections.py`.
