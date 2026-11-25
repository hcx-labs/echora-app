# Upgrading

## To the `/api/v1` release (backend and worker split)

This release replaces the single analysis service with a backend (the API) and workers, and moves
every API route under `/api/v1`. There are no compatibility routes, so an existing deployment
needs these changes at the same time as the new images. No data is reprocessed and the database
schema does not change.

1. **OIDC redirect URI.** Set `OIDC_REDIRECT_URI` to `<site>/api/v1/auth/oidc/callback` and
   register the same URI with the identity provider. With the old path nobody can sign in.
2. **Navidrome plugin.** Install the new plugin build and set its `apiUrl` to
   `<site>/api/v1/plugin`. Until then every plugin call (similar songs, radio, lyrics) gets a 404.
3. **`ECHORA_MEDIA_BASE`.** It is now an origin, and empty means the same origin. Remove an old
   `/api` value, or covers, streams and motion artwork request `/api/api/v1/...` and fail.
4. **Reverse proxies.** Route `/api/v1/` to the backend with the path unchanged (no prefix
   stripping).
5. **Compose.** The `analysis` service is now `backend`. Bring the stack up with
   `docker compose up -d --build --remove-orphans`; otherwise the old `analysis` container keeps
   port 8000 and the new backend cannot start. The web container reads `BACKEND_URL`
   (was `ANALYSIS_URL`).
6. **Images.** `echora-analysis-gpu` and its CPU variant are replaced by `echora-backend`,
   `echora-backend-cpu`, `echora-worker` and `echora-worker-cpu`. Deployments outside Compose
   need the new names. Modal runs `echora-worker`; the Modal app keeps the name
   `echora-analysis`.
7. **Retired settings.** `FA_KARA_ALIGNER` and `FA_KARA_AUDIO_INPUT` are ignored (the aligner
   always uses the full mix), and `ECHORA_LANGUAGE_BACKFILL_BATCH_SIZE` no longer exists. Other
   old setting names keep working and are listed as deprecated in `configs/runtime.env.example`.
8. **Backfill routes.** The lyrics, voice, karaoke and recordings backfills are now
   administrator-only.
