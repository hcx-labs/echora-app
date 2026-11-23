# External AI backend foundation

Apply Alembic revision `0051_external_ai_translations` through the normal migration
process. No configuration row is required: reads default to disabled. Downgrade
drops configuration and translations (including encrypted keys) but never source
lyrics. Back up these new tables before downgrading if their contents are needed.

## Settings API

`GET /settings/external-ai` and `PUT /settings/external-ai` require an authenticated
administrator using the existing session cookie. PUT replaces configuration:

```json
{
  "enabled": false,
  "url": "http://localhost:8000/v1",
  "model": "local-model",
  "prompt": "Translate each lyric line faithfully, preserving meaning, tone and line boundaries. Do not add commentary.",
  "language_pairs": [{"source": "ja", "target": "en"}],
  "api_key": "replacement key"
}
```

Omit/null `api_key` to preserve, send an empty string to clear, or supply a new
value to replace. Encryption uses the existing credential Fernet key via `_cipher`;
no new secret configuration is introduced. Responses expose only `has_key`, never
plaintext or ciphertext. PUT validation errors are deliberately generic to avoid
Pydantic echoing secret-bearing input. Language tags are trimmed, lowercased and
underscore-to-hyphen normalized; duplicate pairs and same-language pairs fail.
Omitting prompt restores the built-in default; custom prompts must be nonblank.

The URL is an API base; `/chat/completions` is appended. Local/private HTTP
endpoints are intentionally allowed. Administrators are trusted to control egress,
including destinations on internal networks. Use network policy where this trust
is inappropriate. HTTP can expose both lyrics and keys: prefer HTTPS off-host.
Changing endpoint while preserving a key will send that key to the new endpoint.
Userinfo, query strings, fragments, whitespace and non-HTTP(S) schemes are rejected.
Requests use the official `openai` Python SDK, pinned to 3.19.2, with `max_completion_tokens`. SDK retries are disabled so future workers can control retry budgets and cancellation. Keyless local endpoints omit Authorization, and SDK organization/project/admin-key environment defaults are not used. Raw response streaming retains the size limit and strict JSON validation. The client disables redirects and environment proxies, uses finite connect/pool
(5s) and read/write (30s) timeouts, checks a 60s budget between received chunks,
and caps input/output size. A blocking read may last until its read timeout after
the budget; this is not a strict wall-clock cancellation mechanism.

## Claude provider

Revision `0063_external_ai_provider` adds `provider`: `openai` (default, the
OpenAI-compatible endpoint above) or `anthropic`. With `anthropic`, translation,
transcription refinement and the motion artwork writer call Claude through the official
`anthropic` Python SDK, pinned to 2.9.0 (`claude_client.py`). The URL is optional and
defaults to `https://api.anthropic.com`; the model is a Claude model ID such as
`claude-opus-5-5`. A key is required: the client never falls back to `ANTHROPIC_API_KEY`
or other environment or profile credentials. Replies use structured outputs (a JSON schema
per feature), then go through the same validation as the OpenAI path. SDK retries, redirects
and environment proxies stay disabled. Claude replies are not streamed, so each request has a
single deadline (translation 180s, refinement 240s, motion writer 120s). A refusal or a reply
cut off at `max_tokens` fails with `refusal` or `incomplete_response` and the caller keeps its
usual fallback. Changing provider keeps the stored key, so replace the key as well.

## Translation contract and storage

`lyric_translation.translate` is opt-in, refuses disabled settings/unconfigured
pairs, and accepts ordered source lines. It sends them as untrusted JSON with
zero-based IDs, requesting JSON-only output. Exact ID coverage is required;
duplicates, unknown IDs, incorrect types, extra line fields, multiline outputs,
empty translations of nonblank sources, malformed JSON, refusals, tools and
non-`stop` completion reasons fail with safe error codes. Outputs are reordered by
ID. There are no automatic retries or tools. Prompt instructions reduce injection
risk but cannot guarantee linguistic fidelity; output is always inert data.

A song already in a pair's target language (English, say) can still have lines in
other languages: an Urdu verse in an English song. When at least two lines are
foreign by the karaoke aligner's per-line rule (`label_rules.line_language`, which
also catches romanised Urdu, Hindi or Korean that fastText cannot name), or fastText
is at least 0.7 sure a line is in a language some pair translates from, the whole
song is sent with `mixed=True`, which asks the model to translate only the lines in
another language. Only those lines keep a translation, and only where the model
changed them; the rest are stored empty and not shown. These are stored with source
language `mul`.

`translation_storage` persists settings atomically and upserts translations by
track, source/target pair, and provenance (`ai` versus `provider`). Each translation
contains mapped lines, a checksum of the exact ordered source text, model, prompt
hash, status and timestamps. AI writes cannot replace imported provider results.
`load_translations` marks mismatched source checksums stale. This requires callers
to pass the same canonical resolved source-line list used for translation. Original
lyrics and karaoke timing are never modified.

## Integrated behavior and remaining work

The Settings UI uses separate admin-only GET/PUT routes:

- `/settings/external-ai/endpoint`: enabled, URL, key replacement and redacted key status.
- `/settings/external-ai/translation`: model, prompt and language pairs.

The combined route remains available for compatibility. Section writes update only their own columns. API enablement requires a URL; translation execution also requires a model and language pairs.

Sync invokes translation after lyrics processing. Current translations are reused by source checksum; changing the prompt alone does not regenerate them. The authenticated track lyrics API includes ready translations, which fullscreen renders alongside source text without translated karaoke timing.

`DELETE /settings/external-ai/translations` requires admin access and refuses deletion while sync or lyrics jobs remain active. It deletes translations across the instance, not original lyrics. The UI uses an application confirmation dialog.

Remaining: separate translated embeddings and curation retrieval, per-request retries/rate limits, safe persisted failure details, and translation-specific batch failure handling. Currently a translation failure increments the summary failure count, which can cause the worker to retry the whole batch.

Tests include `test_external_ai.py`, `test_translation_pipeline.py`, `test_translation_storage.py` and `test_browse_filters.py`. PostgreSQL storage tests require `TEST_DATABASE_URL` and use an isolated schema. Migrations 0051 through 0054 add translation storage, update unchanged default prompts, and create Browse indexes concurrently.
