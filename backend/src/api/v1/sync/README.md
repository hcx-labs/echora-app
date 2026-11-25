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
| `POST /connections/{id}/backfills/{recordings,lyrics,voice,karaoke}` | `backfill_*.py` | Run one operation's stages over the connection (administrators) |
| `GET`, `POST /semantic-fusion` | `semantic_fusion_*.py` | Fused vector counts; rebuild them |
| `GET`, `POST /audio-profiles` | `audio_profiles_*.py` | Audio profile coverage; rebuild them |

### Backfills (operator tools)

A backfill runs only some of a sync's per-song stages over the connection's songs, skipping the
catalog scan, audio embeddings and the other stages. Each stage still only processes songs that
are missing or out of date, so a backfill never does work a full sync would not; it is the cheap
way to redo one stage, for example after a new karaoke model revision.

| Backfill | Job kind | Stages |
| --- | --- | --- |
| `recordings` | `recordings_backfill` | recording fingerprints |
| `lyrics` | `lyrics_backfill` | lyrics, translations, themes, karaoke |
| `karaoke` | `karaoke_backfill` | lyrics, karaoke |
| `voice` | `voice_backfill` | voice |

They are for administrators (403 otherwise), on a connection they own, and the web app has no
button for them. One job per kind and connection is active at a time; it shows on the sync
page like any other job. With an administrator's session cookie (`echora_session`, from the
browser) and the connection id (`navidrome_connection_id` in `GET /api/v1/auth/me`):

```sh
curl -X POST -b "echora_session=$SESSION" \
  "https://echora.example/api/v1/sync/connections/$CONNECTION_ID/backfills/karaoke"
```

`GET /connections/{id}/status` scans the whole Navidrome library, and heals renumbered sources
(`relink_renumbered_sources`) on the way, so it is not free of writes.

Helpers live in `backend/src/sync/connections.py`.
