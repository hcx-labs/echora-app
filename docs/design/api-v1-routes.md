# API v1 route map

Every route of `services/analysis` and its `/api/v1` replacement (hard cutover, no legacy paths;
see [analysis-architecture.md](analysis-architecture.md#api-v1)). The second column is the
endpoint's file under `backend/src/api/v1/`.

Paths keep their resource names; what changes is the `/api/v1` prefix and the category each
route sits in. Methods and bodies are unchanged unless noted.

## system

| Old | New | File |
| --- | --- | --- |
| `GET /health` | `GET /api/v1/system/health` | `system/health.py` |
| `GET /capabilities` | `GET /api/v1/system/capabilities` | `system/capabilities.py` |

## auth

| Old | New | File |
| --- | --- | --- |
| `GET /auth/oidc/status` | `GET /api/v1/auth/oidc/status` | `auth/oidc_status.py` |
| `GET /auth/oidc/start` | `GET /api/v1/auth/oidc/start` | `auth/oidc_start.py` |
| `GET /auth/oidc/callback` | `GET /api/v1/auth/oidc/callback` | `auth/oidc_callback.py` |
| `POST /auth/logout` | `POST /api/v1/auth/logout` | `auth/logout.py` |
| `GET /auth/me` | `GET /api/v1/auth/me` | `auth/me.py` |
| `GET /auth/session` | `GET /api/v1/auth/session` | `auth/session.py` |
| `POST /auth/session/activity` | `POST /api/v1/auth/session/activity` | `auth/session_activity.py` |

The OIDC callback moves: `ECHORA_OIDC_REDIRECT_URI` and the identity provider's registered
redirect URI change with the release (`…/api/v1/auth/oidc/callback`).

## settings

| Old | New | File |
| --- | --- | --- |
| `GET /settings` | `GET /api/v1/settings` | `settings/get.py` |
| `PUT /users/me/preferences/onboarding` | `PUT /api/v1/settings/onboarding` | `settings/onboarding.py` |
| `PUT /settings/profile` | `PUT /api/v1/settings/profile` | `settings/profile.py` |
| `PUT /settings/timezone` | `PUT /api/v1/settings/timezone` | `settings/timezone.py` |
| `PUT /settings/navidrome` | `PUT /api/v1/settings/navidrome` | `settings/navidrome.py` |
| `PUT /settings/lastfm` | `PUT /api/v1/settings/lastfm` | `settings/lastfm.py` |
| `DELETE /settings/lastfm` | `DELETE /api/v1/settings/lastfm` | `settings/lastfm_disconnect.py` |
| `GET /settings/runtime` | `GET /api/v1/settings/runtime` | `settings/runtime.py` |
| `PUT /settings/models/karaoke` | `PUT /api/v1/settings/models/karaoke` | `settings/models_karaoke.py` |
| (new) | `PUT /api/v1/settings/models/karaoke/realign` | `settings/models_karaoke_realign.py` |
| `PUT /settings/models/huggingface` | `PUT /api/v1/settings/models/huggingface` | `settings/models_huggingface.py` |
| `PUT /settings/models/transcription` | `PUT /api/v1/settings/models/transcription` | `settings/models_transcription.py` |
| `PUT /settings/models/hum` | `PUT /api/v1/settings/models/hum` | `settings/models_hum.py` |
| `GET /settings/oidc` | `GET /api/v1/settings/oidc` | `settings/oidc.py` |
| `PUT /settings/oidc/policy` | `PUT /api/v1/settings/oidc/policy` | `settings/oidc_policy.py` |
| `POST /settings/oidc/allowed-emails` | `POST /api/v1/settings/oidc/allowed-emails` | `settings/oidc_allow.py` |
| `DELETE /settings/oidc/allowed-emails/{email}` | `DELETE /api/v1/settings/oidc/allowed-emails/{email}` | `settings/oidc_disallow.py` |
| `PATCH /settings/oidc/users/{user_id}` | `PATCH /api/v1/settings/oidc/users/{user_id}` | `settings/oidc_user.py` |
| `GET, PUT /settings/external-ai` | `GET, PUT /api/v1/settings/external-ai` | `settings/external_ai.py`, `settings/external_ai_update.py` |
| `GET, PUT /settings/external-ai/{endpoint,translation,transcription,themes}` | same under `/api/v1` | `settings/external_ai_{section}.py`, `settings/external_ai_{section}_update.py` |
| `DELETE /settings/external-ai/translations` | `DELETE /api/v1/settings/external-ai/translations` | `settings/external_ai_translations.py` |
| `GET, PUT /settings/external-processing` | same under `/api/v1` | `settings/external_processing*.py` |
| `POST /settings/external-processing/prepare` | same under `/api/v1` | `settings/external_processing_prepare.py` |
| `GET /settings/external-processing/availability` | same under `/api/v1` | `settings/external_processing_availability.py` |
| `GET, PUT /settings/motion-artwork` | same under `/api/v1` | `settings/motion_artwork*.py` |
| `PUT /settings/motion-artwork/location` | same under `/api/v1` | `settings/motion_artwork_location.py` |
| `GET /settings/motion-artwork/status` | same under `/api/v1` | `settings/motion_artwork_status.py` |
| `DELETE /settings/motion-artwork/loops` | same under `/api/v1` | `settings/motion_artwork_loops.py` |
| `GET, PUT /settings/integrations/navidrome` | same under `/api/v1` | `settings/plugin*.py` |
| `POST /settings/integrations/navidrome/keys` | same under `/api/v1` | `settings/plugin_keys.py` |
| `DELETE /settings/integrations/navidrome/keys/{key_id}` | same under `/api/v1` | `settings/plugin_key_delete.py` |

## sync

Connections, imports, full syncs and the library-wide rebuilds a user starts.

The four backfills are now for administrators only (any signed-in user before).

| Old | New | File |
| --- | --- | --- |
| `POST /navidrome/discover` | `POST /api/v1/sync/discover` | `sync/discover.py` |
| `POST /ingest/navidrome` | `POST /api/v1/sync/ingest` | `sync/ingest.py` |
| `GET /navidrome/connections/{id}/sample` | `GET /api/v1/sync/connections/{id}/sample` | `sync/sample.py` |
| `POST /navidrome/connections/{id}/import` | `POST /api/v1/sync/connections/{id}/import` | `sync/import_sample.py` |
| `GET /navidrome/connections/{id}/catalog` | `GET /api/v1/sync/connections/{id}/catalog` | `sync/catalog.py` |
| `POST /navidrome/connections/{id}/sync` | `POST /api/v1/sync/connections/{id}` | `sync/start.py` |
| `GET /navidrome/connections/{id}/sync/status` | `GET /api/v1/sync/connections/{id}/status` | `sync/status.py` |
| `POST /navidrome/connections/{id}/recordings/backfill` | `POST /api/v1/sync/connections/{id}/backfills/recordings` | `sync/backfill_recordings.py` |
| `POST /navidrome/connections/{id}/lyrics/backfill` | `POST /api/v1/sync/connections/{id}/backfills/lyrics` | `sync/backfill_lyrics.py` |
| `POST /navidrome/connections/{id}/voice/backfill` | `POST /api/v1/sync/connections/{id}/backfills/voice` | `sync/backfill_voice.py` |
| `POST /navidrome/connections/{id}/lyrics/karaoke/backfill` | `POST /api/v1/sync/connections/{id}/backfills/karaoke` | `sync/backfill_karaoke.py` |
| `POST /library/semantic-fusion/rebuild` | `POST /api/v1/sync/semantic-fusion` | `sync/semantic_fusion_rebuild.py` |
| `GET /library/semantic-fusion/status` | `GET /api/v1/sync/semantic-fusion` | `sync/semantic_fusion_status.py` |
| `POST /library/audio-profiles/rebuild` | `POST /api/v1/sync/audio-profiles` | `sync/audio_profiles_rebuild.py` |
| `GET /library/audio-profiles/status` | `GET /api/v1/sync/audio-profiles` | `sync/audio_profiles_status.py` |

## library

| Old | New | File |
| --- | --- | --- |
| `GET /library/tracks` | `GET /api/v1/library/tracks` | `library/tracks.py` |
| `GET /library/tracks/{id}/info` | `GET /api/v1/library/tracks/{id}/info` | `library/track_info.py` |
| `GET /library/tracks/{id}/similar` | `GET /api/v1/library/tracks/{id}/similar` | `library/track_similar.py` |
| `GET /library/tracks/{id}/recording-group` | same under `/api/v1` | `library/track_recording_group.py` |
| `GET /library/tracks/{id}/audio-descriptors` | same under `/api/v1` | `library/track_audio_descriptors.py` |
| `GET /library/tracks/{id}/audio-quality` | same under `/api/v1` | `library/track_audio_quality.py` |
| `GET /library/tracks/{id}/audio-profile` | same under `/api/v1` | `library/track_audio_profile.py` |
| `GET /library/tracks/{id}/waveform` | same under `/api/v1` | `library/track_waveform.py` |
| `GET /library/tracks/{id}/visual-features` | same under `/api/v1` | `library/track_visual_features.py` |
| `GET /library/tracks/{id}/motion-artwork` | same under `/api/v1` | `library/track_motion_artwork.py` |
| `GET /library/tracks/{id}/lyrics` | same under `/api/v1` | `library/track_lyrics.py` |
| `PUT /library/tracks/{id}/lyrics` | same under `/api/v1` | `library/track_lyrics_update.py` |
| `PUT /library/tracks/{id}/lyrics/status` | same under `/api/v1` | `library/track_lyrics_status.py` |
| `PUT /library/tracks/{id}/lyrics/transcription-language` | same under `/api/v1` | `library/track_transcription_language.py` |
| (new) | `GET /api/v1/library/tracks/{id}/processing` | `library/track_processing.py` |
| (new) | `PUT, DELETE /api/v1/library/tracks/{id}/processing/{pipeline}` | `library/track_processing_resync.py` |
| `GET /library/lyrics/status` | `GET /api/v1/library/lyrics/status` | `library/lyrics_status.py` |
| `GET /library/artists/profile` | same under `/api/v1` | `library/artist_profile.py` |
| `GET /library/artists/similar` | same under `/api/v1` | `library/artists_similar.py` |
| `GET /library/map` | same under `/api/v1` | `library/map.py` |
| `GET /library/community-snapshots` | same under `/api/v1` | `library/community_snapshots.py` |
| `GET /library/community-snapshots/{id}` | same under `/api/v1` | `library/community_snapshot.py` |
| `GET /library/filter-options` | same under `/api/v1` | `library/filter_options.py` |
| `GET /library/facets` | same under `/api/v1` | `library/facets.py` |
| `GET /library/top-tracks` | same under `/api/v1` | `library/top_tracks.py` |
| `GET /library/concepts` | same under `/api/v1` | `library/concepts.py` |
| `POST /library/concepts` | same under `/api/v1` | `library/concept_create.py` |
| `POST /library/concepts/preview` | same under `/api/v1` | `library/concept_preview.py` |

## search

| Old | New | File |
| --- | --- | --- |
| `POST /library/hum/search` | `POST /api/v1/search/hum` | `search/hum.py` |
| `GET /library/hum/index` | `GET /api/v1/search/hum/index` | `search/hum_index_status.py` |
| `POST /library/hum/index` | `POST /api/v1/search/hum/index` | `search/hum_index_build.py` |
| `GET /library/recording/status` | `GET /api/v1/search/recording/status` | `search/recording_status.py` |
| `POST /library/recording/search` | `POST /api/v1/search/recording` | `search/recording.py` |
| `GET /library/recording/search/{job_id}` | `GET /api/v1/search/recording/{job_id}` | `search/recording_result.py` |
| `DELETE /library/recording/search/{job_id}` | `DELETE /api/v1/search/recording/{job_id}` | `search/recording_cancel.py` |
| `GET, POST /library/recording/calibration/samples` | `GET, POST /api/v1/search/recording/calibration/samples` | `search/calibration_samples*.py` |
| `DELETE /library/recording/calibration/samples/{id}` | same under `/api/v1/search/recording` | `search/calibration_sample_delete.py` |
| `POST /library/concepts/lens` | `POST /api/v1/search/concepts` | `search/concepts.py` |

## curations

| Old | New | File |
| --- | --- | --- |
| `GET /library/curations` | `GET /api/v1/curations` | `curations/list.py` |
| `POST /library/curations` | `POST /api/v1/curations` | `curations/create.py` |
| `PUT /library/curations/{id}` | `PUT /api/v1/curations/{id}` | `curations/replace.py` |
| `PATCH /library/curations/{id}` | `PATCH /api/v1/curations/{id}` | `curations/update.py` |
| `DELETE /library/curations/{id}` | `DELETE /api/v1/curations/{id}` | `curations/delete.py` |
| `POST /library/curations/{id}/refresh` | `POST /api/v1/curations/{id}/refresh` | `curations/refresh.py` |
| `POST /library/curations/preview` | `POST /api/v1/curations/preview` | `curations/preview.py` |
| `POST /library/journeys/preview` | `POST /api/v1/curations/journeys/preview` | `curations/journey_preview.py` |

## jobs

| Old | New | File |
| --- | --- | --- |
| `GET /jobs` | `GET /api/v1/jobs` | `jobs/list.py` |
| `GET /jobs/{id}` | `GET /api/v1/jobs/{id}` | `jobs/get.py` |
| `POST /jobs/{id}/cancel` | `POST /api/v1/jobs/{id}/cancel` | `jobs/cancel.py` |
| `POST /jobs/{id}/dismiss` | `POST /api/v1/jobs/{id}/dismiss` | `jobs/dismiss.py` |
| `GET /jobs/{id}/batches` | `GET /api/v1/jobs/{id}/batches` | `jobs/batches.py` |

## media

| Old | New | File |
| --- | --- | --- |
| `GET /navidrome/connections/{id}/stream/{song_id}` | `GET /api/v1/media/connections/{id}/stream/{song_id}` | `media/stream.py` |
| `GET /navidrome/connections/{id}/cover/{cover_id}` | `GET /api/v1/media/connections/{id}/cover/{cover_id}` | `media/cover.py` |
| `GET /motion-artwork/{artwork_id}.mp4` | `GET /api/v1/media/motion-artwork/{artwork_id}.mp4` | `media/motion_artwork.py` |
| `POST /navidrome/scrobble` | `POST /api/v1/media/scrobble` | `media/scrobble.py` |

## plugin

The Navidrome plugin's base URL becomes `…/api/v1/plugin` (users update the plugin setting with
the new plugin build).

| Old | New | File |
| --- | --- | --- |
| `POST /integrations/navidrome/v1/similar-tracks` | `POST /api/v1/plugin/similar-tracks` | `plugin/similar_tracks.py` |
| `POST /integrations/navidrome/v1/similar-artists` | `POST /api/v1/plugin/similar-artists` | `plugin/similar_artists.py` |
| `POST /integrations/navidrome/v1/artist-radio` | `POST /api/v1/plugin/artist-radio` | `plugin/artist_radio.py` |
| `POST /integrations/navidrome/v1/sonic-path` | `POST /api/v1/plugin/sonic-path` | `plugin/sonic_path.py` |
| `POST /integrations/navidrome/v1/lyrics` | `POST /api/v1/plugin/lyrics` | `plugin/lyrics.py` |
