# Library (`/api/v1/library`)

The signed-in user's songs and what Echora knows about them: browsing, per-song analysis
(descriptors, waveform, visualizer features, audio profiles, recording groups, motion artwork),
lyrics and their editing, artists, the library map and concepts. Every endpoint needs a session;
per-song endpoints answer 404 for songs outside the user's library (except `similar` and
`recording-group`, which only need a session).

| Endpoint | File |
| --- | --- |
| `GET /tracks` | `tracks.py` (search, artist/album, browse filters from `backend/library/browse_filters.py`, sort, paging) |
| `GET /tracks/{id}/info` | `track_info.py` (facts from `backend/library/track_info.py`) |
| `GET /tracks/{id}/similar` | `track_similar.py` |
| `GET /tracks/{id}/recording-group` | `track_recording_group.py` |
| `GET /tracks/{id}/audio-descriptors` | `track_audio_descriptors.py` |
| `GET /tracks/{id}/audio-quality` | `track_audio_quality.py` |
| `GET /tracks/{id}/audio-profile` | `track_audio_profile.py` |
| `GET /tracks/{id}/waveform` | `track_waveform.py` |
| `GET /tracks/{id}/visual-features` | `track_visual_features.py` |
| `GET /tracks/{id}/motion-artwork` | `track_motion_artwork.py` (the video is `/media/motion-artwork/<id>.mp4`) |
| `GET /tracks/{id}/lyrics` | `track_lyrics.py` |
| `PUT /tracks/{id}/lyrics` | `track_lyrics_update.py` |
| `PUT /tracks/{id}/lyrics/status` | `track_lyrics_status.py` |
| `PUT /tracks/{id}/lyrics/transcription-language` | `track_transcription_language.py` |
| `GET /lyrics/status` | `lyrics_status.py` |
| `GET /artists/profile` | `artist_profile.py` (stores the profile in `artist_profiles`) |
| `GET /artists/similar` | `artists_similar.py` |
| `GET /map` | `map.py` (computed by `backend/library/community_map.py`, stored as a community snapshot) |
| `GET /community-snapshots` | `community_snapshots.py` |
| `GET /community-snapshots/{id}` | `community_snapshot.py` |
| `GET /filter-options` | `filter_options.py` |
| `GET /facets` | `facets.py` |
| `GET /top-tracks` | `top_tracks.py` (Last.fm, or Navidrome play counts cached for 5 minutes) |
| `GET /concepts` | `concepts.py` |
| `POST /concepts` | `concept_create.py` |
| `POST /concepts/preview` | `concept_preview.py` (MuQ-MuLan corpus from `backend/search/corpora.py`) |

Errors: `NotFoundError` (404) for songs, artists, profiles and snapshots that are not there,
`InvalidQueryError` (422) for unknown parameters, `ConflictError` (409) when a corpus has no
embeddings or AI lyrics are off, `IntegrationError` (502) when Last.fm or Navidrome fails.
