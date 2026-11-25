# Runtime settings

Deployment settings are strict Pydantic config classes, one per domain, each in a `config.py` next to the code it configures (for example `worker/src/kara/config.py`). Every field is the environment variable `<prefix><FIELD>`; older names are still read and listed as deprecated in the catalog. `<Config>.load()` validates and caches a class once per process. Restart the API and workers after changing configuration.

Environment variables override Python defaults. Existing names such as `DATABASE_URL`, `MUQ_REVISION`, and `COOKIE_SECURE` remain supported. New operational controls generally use `ECHORA_` names. Units appear in each setting name. Worker command-line arguments override their environment defaults.

The application does not load dotenv files implicitly. Docker Compose passes the project `.env` to the API and workers through `env_file`, with Compose's explicit `environment` entries taking precedence. This optional-file syntax requires Docker Compose 2.24 or newer. For direct Python execution, export the variables before starting the process.

Start with `configs/env.example`. The complete, commented catalog is `configs/runtime.env.example`. Copy only the overrides you need into `.env` and uncomment them. Regenerate the catalog without reading environment values:

```sh
PYTHONPATH=services/analysis/src python scripts/export-runtime-settings.py  # with core and worker installed (pip install -e core -e worker) > configs/runtime.env.example
```

Browser polling uses the same defaults, exported to `web/components/runtime/runtimeConfig.defaults.json`. The browser loads overrides from the API once after mount. During an API outage it keeps the generated defaults. Regenerate the fallback after changing those settings:

```sh
PYTHONPATH=services/analysis/src python scripts/export-runtime-settings.py  # with core and worker installed (pip install -e core -e worker) --ui > web/components/runtime/runtimeConfig.defaults.json
```

Secrets are blank in that catalog and excluded from config representations and serialization. Do not log a config object or publish its complete contents. `/settings/runtime` returns only an explicit allowlist of browser polling intervals and request timeouts.

## Sessions

Echora stores its own opaque session token in an HttpOnly, SameSite Strict cookie. The database stores only its SHA-256 hash. OIDC authenticates sign-in; provider access-token expiry does not determine the app session lifetime.

| Environment variable | Default |
| --- | --- |
| `ECHORA_SESSION_IDLE_TIMEOUT_SECONDS` | 604800, seven days |
| `ECHORA_SESSION_ABSOLUTE_TIMEOUT_SECONDS` | 2592000, thirty days |
| `ECHORA_SESSION_RENEW_INTERVAL_SECONDS` | 3600, one hour |
| `ECHORA_SESSION_ACTIVITY_CHECK_SECONDS` | 60 |
| `ECHORA_SESSION_ACTIVITY_WINDOW_SECONDS` | 300 |
| `ECHORA_SESSION_REQUEST_TIMEOUT_SECONDS` | 10 |
| `ECHORA_SESSION_TRUSTED_ORIGINS` | Empty, additional comma-separated frontend origins |

The client reports recent interaction or advancing playback to `POST /auth/session/activity`. Ordinary requests, including job polling and `GET /auth/session`, do not renew sessions. Renewal moves the idle deadline but never the absolute deadline. Hourly throttling means idle expiry is measured from the last recorded renewal, potentially almost one hour before the latest interaction.

The activity endpoint requires `X-Echora-Activity: 1` and an approved `Origin`. Approved origins come from the configured OIDC callback and post-login URLs, `ECHORA_CORS_ORIGINS`, and `ECHORA_SESSION_TRUSTED_ORIGINS`. Set the latter for additional frontend URLs. Wildcards and missing or `null` request origins are rejected. The API never trusts forwarded host headers to establish renewal origins.

A PostgreSQL row lock serializes renewals across tabs and processes. Authentication rejects expired, deleted, and blocked sessions. Renewals only update existing rows. Responses have `Cache-Control: no-store`; renewed cookies use the database deadline. `COOKIE_SECURE=true` remains required for HTTPS deployments.

Migration `0050_session_activity` preserves existing sessions' original expiration as their absolute limit. Those sessions cannot gain a longer lifetime through this migration. Sign in again to receive the new policy. Absolute deadlines are persisted at login, so changing the configured maximum does not rewrite existing sessions.

## Other controls

Settings cover database pools, worker timing and retries, provider requests, cache budgets, preprocessing locks, subprocess deadlines, inference batch sizes, recording quotas, and retention. New recording and calibration rows receive configured expiry values explicitly rather than relying on historical SQL defaults. Existing rows retain their recorded deadlines.

`ECHORA_MODAL_IMAGE` is the base image for analysis on Modal ([docs/modal-compute.md](modal-compute.md)). Release images set it at build time to the GPU analysis image of their own version; override it only to test a different image. The Modal token itself is configured in Settings → External processing, not in the environment.

Model IDs and revisions preserve their previous defaults. Compose intentionally enables a pinned MOSS model while direct Python defaults leave it disabled. Do not uncomment blank MOSS entries in the catalog unless you intend to disable transcription. Settings do not replace database-backed user preferences or curation schedules.

Embedding dimensions, sample rates, calibrated matching rules, preprocessing recipes, and UI geometry remain code or versioned model configuration. `ECHORA_JOB_ID` is per-job state. `TMPDIR`, `PYTHONPATH`, `HF_HOME`, and `NUMBA_CACHE_DIR` remain process or third-party library environment. Infrastructure settings such as container ports, Redis memory limits, and health checks remain in deployment files.

## Tests

Tests that change environment variables must call `core.config.reset_config_cache()` before reading them and after the test. Pure model utilities can load settings without a database or authentication secrets; database entry points still require a database URL. Invalid values fail validation without echoing the supplied input in the rendered error.
