# Echora architecture overhaul

Status: proposed, 2026-11-25.

## Why

The analysis service (`services/analysis/src/echora_analysis`) is 25,600 lines in 95 flat modules,
and it is three programs in one package:

- an HTTP API: a 4,700-line `main.py` with about 100 routes, sessions and OIDC;
- interactive search: hum, recording, text and theme search;
- background processing: sync, import and backfill batches, run by workers.

Their boundaries are not visible in the code.

- **Workers import the API.** Pipelines import `main` for helpers (`main._load_connection`,
  `main._attach_user_library`).
- **One settings object.** The single 466-line `Settings` mixes sessions, OIDC, worker timing and
  every model knob. Modal receives a hand-kept field list (`PROCESSING_FIELDS`) and adopts it by
  writing environment variables.
- **No common pipeline shape.** Every pipeline is a `backfill_*` function with its own signature,
  progress shape, summary keys and failure handling. `analysis_jobs.execute` hard-codes which
  pipelines an operation runs.
- **Search mixed with processing.** `hum_search.py` builds corpora *and* answers queries;
  `recording_search.py` is both a job and an API.
- **Weak errors and stale tests.** Errors are bare `ValueError` / `RuntimeError`, and tests are
  flat.

The repository layout hides the split too: `apps/web` is the only app under `apps/`,
`plugins/navidrome` the only plugin, and the Python service sits under `services/`.

## Target layout

```
echora/
  web/          Next.js app (was apps/web)
  navidrome/    Navidrome plugin, Go (was plugins/navidrome)
  core/         shared Python library (imports as `core`)
  backend/      FastAPI API and search services (imports as `backend`), image echora-backend
  worker/       sync processing (imports as `worker`), image echora-worker (GPU, Modal)
  database/  configs/  docs/  scripts/  nginx/  testdata/   (unchanged)
```

Each Python project keeps its code in `src/` and its tests in `tests/`, beside `pyproject.toml`.
The code imports under the project's name: `core/src/config.py` is `core.config`, and
`worker/src/kara/` is `worker.kara`. A three-line `setup.py` maps the name to `src/`, for wheels and
for editable installs alike.

Dependencies point one way: `backend -> core`, `worker -> core`. The backend never imports the
worker and the worker never imports the backend. They talk through PostgreSQL: the backend enqueues
jobs and reads results; the worker claims jobs and writes results.

### core (`core`)

What both apps need, and nothing app-specific:

```
core/src/
  config.py        EchoraConfig base class, env naming, secrets, catalog export
  errors.py        EchoraError hierarchy, JobCancelled
  db/              connections, SQLAlchemy models, Alembic migrations
  jobs/            the durable queue: enqueue, claim, lease, progress, finish (both sides use it)
  integrations/    Navidrome client, External AI (OpenAI-compatible, Claude), Last.fm
  media/           audio decoding, prepared-audio cache, Roformer separation
  models/          model snapshots: download, pinning, resident-model cache
  <algorithms>     shared by processing and search, each in its own package:
    melody/        contour extraction (indexing in the worker, queries in the backend)
    embeddings/    MuQ-MuLan, MERT, BGE-M3 encoders (tracks in the worker, text queries in the backend)
    recordings/    recording fingerprint encoder and matcher
    lyrics/        lyrics documents, language detection, storage formats
```

### worker (`worker`)

```
worker/src/
  main.py          worker entry point: `python -m worker analysis|scheduled`
  executor.py      claim supervision, isolated executor process (was worker.py)
  processing/      the processor framework (below)
  operations.py    which processors an operation runs (sync, import, backfills, hum index)
  remote/          Modal session and Modal app
  library/         catalog ingest: identity, source links, fingerprints, waveforms
  embedding/       audio embeddings, descriptors, visual features
  lyrics/          retrieval and lyrics embeddings
  transcription/   AI lyrics (MOSS), timing repair, External AI refinement
  kara/            karaoke alignment (FA-Kara, vendored aligner under kara/vendor/)
  translation/     lyric translation
  themes/          lyric theme descriptions
  voice/           voice classification
  melody/          melody index
  recordings/      recording fingerprints
  profiles/        audio profiles
  fusion/          semantic fusion vectors
  artwork/         motion artwork (ComfyUI)
  curation/        curation refresh (scheduled worker)
```

Each domain package has `README.md` (algorithm and design), `config.py`, `errors.py`,
`processor.py` and the implementation modules it needs.

### backend (`backend`)

```
backend/src/
  app.py           create_app(): middleware, sessions, the v1 router
  api/
    endpoint.py    Endpoint base class
    requests.py    request body models: PostRequest, PutRequest, PatchRequest
    responses.py   shared response models (pages, job snapshots, errors)
    deps.py        dependencies: current user, admin, database session
    v1/
      auth/        oidc_start.py, oidc_callback.py, logout.py, session.py, ...
      library/     tracks.py, track.py, artists.py, ...
      search/      hum.py, recording.py, text.py, themes.py, concepts.py
      jobs/        list.py, get.py, batches.py
      sync/        start.py, ...
      settings/    ...
      curations/   ...
      plugin/      lyrics.py, similar_tracks.py, similar_artists.py, artist_radio.py, sonic_path.py
      media/       stream.py, cover.py
  search/          search services (below)
  sessions/        session store and OIDC
```

## API v1

All routes live under `/api/v1/<category>/<endpoint>`, for example `/api/v1/search/hum` and
`/api/v1/search/recording`. Each endpoint is one class in its own file, inside its category's
folder:

```python
# backend/src/api/v1/search/hum.py
class HumSearchRequest(PostRequest):
    """A hummed or sung recording to match against the user's melody index."""
    limit: int = Field(10, ge=1, le=50)


class HumSearch(Endpoint):
    method = "POST"
    path = "/search/hum"
    request = HumSearchRequest
    response = HumMatches
    summary = "Find songs by humming"

    def handle(self, request: HumSearchRequest, user: User = Depends(current_user)) -> HumMatches:
        return hum_search(user).search(...)
```

- `Endpoint` holds method, path, request and response models, dependencies, status code and OpenAPI
  metadata. A category's `__init__.py` lists its endpoint classes and the v1 router mounts every
  category. Nothing registers routes with decorators on a global `app`.
- Request bodies derive from `RequestModel` (Pydantic, unknown fields rejected, strings stripped):
  - `PostRequest` creates or performs something;
  - `PutRequest` replaces a resource completely, so every field is required unless it has a
    default;
  - `PatchRequest` changes only the fields sent (`model_dump(exclude_unset=True)`).
  Query parameters stay function parameters.
- Responses are Pydantic models, not dicts, so the OpenAPI schema is complete. The web app's
  TypeScript types can be generated from it later.
- Errors: domain exceptions map to HTTP responses in one place (`InvalidQueryError` to 422,
  `NotFoundError` to 404, `IntegrationError` to 502), never with provider messages.

**Cutover:** hard. The web app (`/api/v1/...` through its proxy) and the Navidrome plugin
(`/api/v1/plugin/...`) change in the same release, and the plugin gets a new build. No legacy paths
are kept.

## Sync processing (worker)

### Processor

A processor is one stage of library processing for a batch of songs: karaoke, transcription, audio
embeddings, voice. It declares what it takes, how it is configured and what it produces:

```python
class KaraokeProcessor(Processor[KaraConfig, KaraokeInput, KaraokeOutput]):
    name = "karaoke"
    summary_key = "karaoke"          # key of this stage in the batch summary (UI, retry policy)

    def plan(self, context, batch) -> list[KaraokeInput]:
        """Songs of the batch that need this stage (missing or stale results). Reads only."""

    def open(self, context) -> None:
        """Load models and do other one-off work. Called once, only when plan() found work."""

    def process(self, context, items) -> Iterator[ItemResult[KaraokeOutput]]:
        """Compute, in phases (one resident model each). Yields one result per item, in order."""

    def persist(self, context, item, output) -> None:
        """Store one result as soon as it exists."""

    def close(self) -> None:
        """Release models. Always called after open()."""
```

`Input` and `Output` are frozen Pydantic models. Inputs carry everything a remote container needs
(content digest, lyrics text, language), so the same processor runs locally or on Modal.

The **runner** owns what is copied between pipelines today:

- **Planning first.** If nothing is planned, nothing else runs and no model is loaded.
- **Open and close.** `open()` / `close()` wrap processing, and `close()` also runs on failure and
  cancellation.
- **Per-item failures.** An item that raises `ProcessingError` is recorded as failed and the batch
  continues. Any other exception is a bug: it is logged with the item id, never with its message,
  because provider messages can contain signed URLs.
- **Progress and cancellation.** Progress has one shape (`phase`, `message`, `completed`, `total`,
  `unit`), with a cancellation check between items.
- **Stage summary.** `StageSummary(total, failed, counters)` is rendered with the keys each stage
  already reports (`aligned`, `supplied`, `embedded`, …). The retry policy and the web app keep
  working unchanged.

A processor that needs several models runs them as phases inside `process()` (separation, then
alignment), loading each once per batch, as `docs/batch-phases.md` describes.

### Operations

`operations.py` replaces the `if operation in {...}` chain of `analysis_jobs.execute` with data:

```python
OPERATIONS = {
    "navidrome_sync": Operation(stages=(LIBRARY, PROFILES, LYRICS, TRANSLATION, THEMES, KARAOKE, VOICE, ARTWORK), expands=True),
    "lyrics_backfill": Operation(stages=(LYRICS, TRANSLATION, THEMES, KARAOKE)),
    ...
}
```

A root job plans and expands into batches as today. A batch runs its operation's stages in order
through the runner. Work that must happen before any GPU stage (motion artwork prompts) is a
declared `prepare` hook of its processor, run first.

### Context

`ProcessingContext` replaces the `check`, `report`, `progress=`, `external_ids=` and
`*credentials` arguments threaded through every pipeline:

- `check()` raises `JobCancelled`, a `BaseException`, so `except Exception` never swallows it;
- `report(update)`;
- `location` (`local` / `modal`) and `remote`, the batch's Modal session or `None`;
- `library`, the Navidrome client and library id for the batch, and `user_id`;
- `audio`, the batch's prepared-audio cache.

## Search services (backend)

```python
class HumSearch(SearchService[MelodyConfig, HumQuery, HumMatches]):
    name = "hum"
    persistent = True                 # keep the instance (and its index) between requests

    def index_key(self) -> str:
        """Changes when the index must be rebuilt (the user's corpus revision)."""

    def open(self) -> None:           # load indexes and models
    def search(self, query: HumQuery) -> HumMatches:
```

- **Request and response.** Search services are not scheduled: they receive a request and return a
  response.
- **Open.** `open()` loads indexes or models.
- **Persistent services.** The registry caches a `persistent` service and reuses it until its
  `index_key()` changes, or until the per-class LRU bound evicts it. It is keyed by service and
  scope (the user, for a per-user index).
- **Other services.** A non-persistent service is opened and closed per request.
- **Heavy queries.** Requests with heavy work run in a bounded thread pool, as hum search does
  today.

Services: hum search, recording search, text-to-audio search, themes search, concept search,
plugin ranking (similar tracks, similar artists, artist radio, sonic path).

## Configuration

`core.config.EchoraConfig` is a `pydantic_settings.BaseSettings` subclass. Each domain and
app defines one:

```python
class KaraConfig(ProcessorConfig, env_prefix="ECHORA_KARA_"):
    model_id: str = compute("hcxlabs/echora-kara-700m-muq")
    min_letter_ms: int = compute(40, ge=0, legacy="FA_KARA_MIN_LETTER_MS")
```

- **Environment names.** Every field is the environment variable `<prefix><FIELD>`, with no extra
  code.
- **Old names.** They keep working through `legacy=` until the next release, and the settings
  catalog lists them as deprecated.
- **Secrets.** They are `SecretStr`, excluded from dumps and reprs.
- **Compute fields.** Fields made with `compute()` change model results. A Modal call ships
  `worker.remote.compute.compute_settings()` (each compute config's `compute_values()`), and the
  container calls `adopt_compute_settings()`, which rebuilds each config with
  `with_compute_values()` and installs it with `adopt()`.
- **No hidden globals.** Global state lives on config, context or service objects. The only
  module-level state is declared, typed caches (resident models, persistent search services), each
  behind one accessor.
- **Catalog.** `scripts/export-runtime-settings.py` walks every config class, so
  `configs/runtime.env.example` stays complete.

## Errors

```
EchoraError
├── ConfigurationError          invalid or missing configuration
├── IntegrationError            Navidrome, External AI, Modal, ComfyUI
│   └── RemoteComputeError
├── ProcessingError             one item of one stage failed; the batch continues
│   ├── InputUnavailableError        audio or lyrics could not be read
│   ├── ModelUnavailableError        a model snapshot or file is missing
│   └── <domain errors>         kara.AlignmentError, transcription.TranscriptionError, …
├── SearchError                 a search request failed
│   └── InvalidQueryError
└── NotFoundError                    the requested resource does not exist or is not visible
JobCancelled (BaseException)    cooperative cancellation or a lost claim
```

## Images and deployment

- **`echora-worker`** (from `worker/`): the current GPU analysis image's dependencies (torch,
  Essentia, Roformer, FA-Kara, MOSS, ComfyUI support), Modal client. It runs both worker types
  (`analysis`, `scheduled`). A CPU variant stays for CPU-only installs.
- **`echora-backend`** (from `backend/`): FastAPI, plus the model dependencies that search needs
  (MuQ-MuLan and BGE-M3 text encoders, the recording encoder, melody contour extraction). It runs
  Alembic migrations at startup, as the API does today.
- **`echora-web`**: unchanged, built from `web/`.
- Compose and the release workflow build three images. The homelab chart needs the new image names
  and commands; that change lives in its own repository.

## Tooling

Ruff, configured once at the repository root (`ruff.toml`) and applied to `core`, `backend` and
`worker`:

- format: PEP 8 through Ruff's Black-compatible style, line length 100, double quotes;
- lint: `E`, `W` (pycodestyle, PEP 8), `F` (pyflakes), `I` (PEP 8 import grouping), `N` (PEP 8
  naming), `D` with the PEP 257 convention (module, package and class docstrings required),
  `UP`, `B`, `SIM`, `C4`, `RET`, `PIE`, `PTH`;
- the vendored aligner is research code that follows its upstream: formatted, not linted;
- pre-commit runs `ruff format` and `ruff check`.

## Tests

Each Python app has `tests/` mirroring its package (`worker/tests/kara/test_lines.py`,
`backend/tests/api/v1/search/test_hum.py`):

- unit tests of pure functions;
- processor tests through the runner with a fake context;
- endpoint tests through FastAPI's test client with fake services;
- database tests on an isolated schema of `TEST_DATABASE_URL` through one shared fixture (in core).

Old tests are rewritten for the new modules, not kept as shims.

## Dead code

Code is removed when nothing calls it (verified by search, not only by Vulture), and each PR lists
what it removed. Found so far:

- `audio.full_coverage_windows` and `audio.deterministic_windows`;
- `karaoke_pipeline.apply_adaptive_line_padding` and `karaoke_pipeline._line_key`;
- `jobs.checkpoint`;
- `journeys.multi_stop_targets`;
- `concepts.expand_prompts`;
- `db_models.Track`;
- `hum_search.match_contour`;
- the karaoke vocal-stem input (`FA_KARA_AUDIO_INPUT=vocals`, superseded by the full mix);
- the MMS aligner path (`FA_KARA_ALIGNER=mms`).

Kept for a decision: the `*_backfill` routes (`lyrics`, `voice`, `karaoke`, `recordings`) have no
caller in the web app. They may be operator tools.

## Non-goals

Results do not change. Pipeline revisions, stored formats, job summaries and the database schema
stay identical, so the refactor reprocesses nothing in the library.

## Branches

Each branch is based on the previous one and keeps every test suite green.

1. `refactor/repo-layout`: `apps/web` to `web/`, `plugins/navidrome` to `navidrome/`, paths in
   compose, Dockerfiles, workflows, docs and tooling. Pure moves.
2. `refactor/worker`: `core/` (config, errors, db, jobs, integrations, media, shared algorithms)
   and `worker/` (processor framework, every processing domain, Modal, curation), with tests. The
   API in `services/analysis` uses `core` from here on.
3. `refactor/backend`: `backend/` with API v1 (endpoint classes, request models), the search
   framework and services, sessions. `services/analysis` is removed. The web app and the Navidrome
   plugin move to `/api/v1`.
4. `refactor/images`: `echora-backend` and `echora-worker` images, compose and the release
   workflow.
