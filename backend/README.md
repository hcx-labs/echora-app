# Backend (`backend`)

Echora's HTTP API, served at `/api/v1`. It imports as `backend` (code in `src/`) and runs with
`uvicorn backend.app:create_app --factory`. Every legacy route and its v1 replacement is listed in
[the route map](../docs/design/api-v1-routes.md).

## Layout

- `app.py`: `create_app()`, which sets up the middleware (CORS, the OIDC state cookie), the error
  handlers, the startup checks and the v1 router.
- `config.py`: API settings (`ECHORA_*`): origins, sessions, OIDC, the media cache, browser
  polling and search concurrency.
- `api/`: the endpoint framework.
  - `endpoint.py`: `Endpoint` (one route per class: method, path, request and response models,
    status, dependencies, `handle()`) and `Category` (a folder of endpoints under one prefix).
  - `requests.py`: `PostRequest`, `PutRequest`, `PatchRequest`. Unknown fields are rejected and
    strings are kept as sent.
  - `responses.py`: `ResponseModel` (undeclared fields fail; unset fields are left out) and
    `JobSnapshot`.
  - `deps.py`: `current_user` (from the `echora_session` cookie) and `admin_user`.
  - `errors.py`: maps domain errors to HTTP statuses: `NotFoundError` 404, `ConflictError` 409,
    `InvalidQueryError` 422, `IntegrationError` 502.
  - `navidrome.py`: the Navidrome credentials body that settings and sync share.
- `api/v1/<category>/`: one file per endpoint, plus a README for each category: system, auth,
  settings, sync, library, search, curations, jobs, media and plugin.
- `search/`: search services. `service.py` holds the `SearchService` base, the registry that keeps
  persistent per-scope indexes until `index_key()` changes (LRU-bounded) and the bounded search
  pool. The services are hum search, the per-user embedding corpora, concept search and the
  plugin corpus.
- `sessions/`: the app session store and OpenID Connect sign-in ([README](src/sessions/README.md)).
- `library/`, `media/`, `plugin/`, `settings/`, `sync/`: helpers behind the endpoints of those
  categories.

## Adding an endpoint

1. Add `api/v1/<category>/<name>.py` with an `Endpoint` subclass and its models.
2. List it in the category's `__init__.py` (static paths before parameterised ones).
3. Test it with the test client (`tests/conftest.py` signs in a fake user) in
   `tests/api/v1/<category>/`.

## Tests

`pytest` in this folder. Database tests need `TEST_DATABASE_URL`, a disposable database; they
create and drop their own schema.
