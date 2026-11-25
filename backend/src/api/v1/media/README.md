# Media (`/api/v1/media`)

What the browser plays and shows. Every endpoint needs a session, and a connection id must
belong to the signed-in user (404 otherwise).

| Endpoint | File | What it serves |
| --- | --- | --- |
| `GET /connections/{id}/stream/{song_id}` | `stream.py` | The song, transcoded by `quality` (`original`, `320`, `120`), with range requests |
| `GET /connections/{id}/cover/{cover_id}` | `cover.py` | Cover art at `size` (64 to 1600 px) |
| `GET /motion-artwork/{artwork_id}.mp4` | `motion_artwork.py` | A motion artwork loop for a cover in the user's library (immutable) |
| `POST /scrobble` | `scrobble.py` | Reports playback to Navidrome; 204, or a bare 502 the player ignores |

Covers, and streams requested with `cache=player`, go through the Redis media cache
(`backend/src/media/cache.py`, `ECHORA_MEDIA_CACHE_*`). A stream is cached only when it was
requested from its start and read to the end; cached streams answer range requests themselves.
