# Echora Navidrome plugin

The Echora plugin delegates sonic discovery, metadata recommendations and lyrics
lookup to Echora. It implements Navidrome's provider contracts directly; it does
not require AudioMuse or impersonate AudioMuse's HTTP API.

## Implementation plan and status

1. **Consolidate Settings → Integrations:** Navidrome, Last.fm and the Echora plugin
   settings live together. Implemented.
2. **Account-scoped access:** enabling the integration generates its first key;
   additional labelled keys have configurable expiry and independent revocation.
   Implemented with SHA-256 hashes, one-time secret display and owner checks.
3. **Discovery:** similar tracks, paths, similar artists and artist radio reuse
   existing MuQ, MERT and BGE-M3 representations. Implemented without a new
   embedding index or analysis job.
4. **Native lyrics:** a Lyrics provider serves existing transcriptions, karaoke
   timings and completed translations. Implemented with TTML and LRC export.
5. **Validate and package:** backend credential/isolation/ranking tests, Go adapter
   tests, actual Navidrome lyrics parser tests and the `.ndp` build are provided.
   Installation on a real Navidrome instance and large-library latency measurements
   remain rollout checks.

## Compatibility and identity

The SDK and parser tests are pinned to Navidrome commit
`0e1893530b844898cbf87825fdefd4b13e1ff3cc` (2026-09-29). Use a Navidrome build
containing this commit or equivalent SonicSimilarity, Lyrics and TTML translation
support. Older builds may lack capabilities or enhanced lyrics support; this has
not been validated against every released Navidrome version.

The plugin's configuration is instance-wide. All callers use its configured
Echora key, so they receive the key owner's library scope and ranking preferences.
There is no Navidrome-user identity forwarding. Backend lookups require that
owner's source membership and the selected Navidrome connection's library
namespace; IDs from another server cannot resolve accidentally.

Keys only authorize the plugin API. They cannot access Echora's Settings or
ordinary session routes. Disabling the integration immediately stops key access.
Changing its connection revokes old keys. Existing expiry dates do not change
when the default lifetime is edited; generate a replacement and revoke the old
key to rotate it. Removing the owning account/connection removes its credentials.

## Install

1. Deploy the Echora web and analysis changes and apply Alembic migration
   `0056_navidrome_original_lyrics` (including its predecessor) using the project's
   normal migration workflow. Update the analysis workers as well as the API.
2. Connect the intended Navidrome server in Echora Settings → Integrations,
   enable Echora Navidrome plugin access, and copy the generated key and Plugin
   API URL. That URL must be reachable from Navidrome, including its container.
3. Download the `.ndp` from a `navidrome-plugin-vX.Y.Z` GitHub release (or build
   `dist/echora.ndp` using the instructions below), copy it into Navidrome's
   plugins folder (normally `<DataFolder>/plugins`), rescan, and enable Echora.
4. Set plugin `apiUrl` to the copied URL, such as
   `https://echora.example/analysis/integrations/navidrome/v1`, and `apiKey` to the
   generated key. This is the browser-facing proxy URL, not just the site origin.
5. Put `echora` first in metadata agent order and lyric source priority. Preserve
   the other providers you already use, for example:

   ```toml
   Agents = "echora,lastfm,deezer"
   LyricsPriority = "echora,.ttml,.yaml,.yml,.elrc,.lrc,.srt,.txt,embedded"
   ```

   Environment equivalents are `ND_AGENTS` and `ND_LYRICSPRIORITY`. Restart
   Navidrome after changing its server configuration.

Navidrome discovers the plugin's exported SonicSimilarity capability and supplies
its own public sonic endpoints. Client applications continue talking to Navidrome;
no Echora URL or key belongs in the client. Metadata discovery uses similar songs
by track, similar songs by artist and similar artists. Missing lyrics return an
empty result so Navidrome can try its next source.

The manifest requests configuration and HTTP access, with `requiredHosts: ["*"]`
to support administrator-configured self-hosted addresses. The adapter calls only
the configured URL, never follows redirects and does not expose remote error
bodies or credentials. Use HTTPS when crossing an untrusted network.

## Ranking and lyrics preferences

Musical / lyrical balance defaults to 80% / 20%. Within musical evidence, the
second slider blends MuQ musical character (70%) with MERT acoustic detail (30%).
Similar-track scoring averages available pairwise cosine evidence with these
weights, renormalizes missing components, and clamps the provider score to [0,1].
This is an evidence weight, not a probability or a calibrated match percentage.

Paths interpolate in a normalized weighted embedding space and reuse Echora's
journey selection. If either endpoint lacks lyrics and musical fallback is
selected, the whole path uses audio evidence consistently. Artist discovery uses
Echora's artist facets and representative songs over that weighted space.
Alternate source IDs and recording duplicates are suppressed. Artist limits can
produce fewer results than requested; path endpoints are always retained.

Choose musical fallback or exclusion for missing lyric embeddings. Settings are
stored per Echora account and are not extra parameters in Navidrome's provider
contract. Requests are bounded at 500 songs or 100 similar artists. The first
implementation reads the scoped corpus per discovery request; large libraries
need latency/memory measurement before broader rollout. Existing embeddings are
sufficient, though a future cache or candidate index may improve performance.

Lyrics requests use saved data only and never launch transcription, alignment or
translation. TTML carries line/syllable timing plus translation tracks with their
language and line mapping. Backing vocals the aligner placed on their own stream (an
inline echo, or a whole line) are wrapped in `ttm:role="x-bg"`, the role Apple Music
uses and Navidrome parses as a background agent, so clients can show them apart from
the lead. Only completed translations with a matching source
checksum and matching line IDs are served. Original line IDs are mapped to karaoke
paragraphs, allowing blank stanza separators and trimmed line edges. Karaoke text
differences do not suppress translations. If nonempty line counts differ, export
uses source paragraphs and their timings, retaining karaoke cues for matching lines.
Translated lines identical to their source after trimming surrounding whitespace
are omitted to avoid duplicate display; empty translation tracks are omitted too.
Incomplete predicted syllables fall back to the authoritative line text.

LRC mode provides line timing and separate language variants. Navidrome and
clients determine display support: ordinary text remains available through native
lyrics APIs, while syllable highlighting and explicit translation tracks require
compatible versions/clients. The legacy `getLyrics` endpoint selects main lyrics.

## Sync lyrics guard and original sources

Echora temporarily pauses plugin lyrics while a sync, import, lyrics backfill or
karaoke backfill job is queued, running or waiting for batches on that Navidrome
server. The provider returns `{lyrics: []}`, which tells Navidrome to try the next
source in `LyricsPriority`. Sonic and metadata discovery remain available.
Ensure the priority list includes the original sources you want after `echora`.
During the pause, other Navidrome clients receive those fallback sources too.

The pause is derived from the durable queue rather than toggling the saved
preference. It covers overlapping jobs, including other Echora accounts using the
same configured Navidrome URL, and stays in place across worker restarts/retries.
The UI shows PAUSED and prevents manual re-enabling. Once all matching jobs are
terminal (complete, partial, failed or cancelled), the saved preference takes
effect automatically: previously enabled lyrics resume; previously disabled
lyrics stay off. A worker outage keeps lyrics paused until queued jobs are
completed, recovered or cancelled. Different URL aliases for the same Navidrome
instance must be consolidated to one connection URL for this scope to match.

Each actual provider retrieval archives the Navidrome response separately in
`navidrome_lyrics_sources`, keyed by source library and song ID. It preserves the
original structured tracks, language variants and timing cues independently of
manual overrides, AI transcripts, karaoke and translations in Echora. The latest
retrieval status and the last response with available main lyrics are both kept,
so an unavailable result does not erase the previous original. Existing data is
archived on its next retrieval; this does not retroactively reconstruct provenance
already overwritten before the guard existed.

## API contracts

All routes are POST under `/analysis/integrations/navidrome/v1` through Echora's
web proxy (analysis service path: `/integrations/navidrome/v1`). They require
`Authorization: Bearer <key>` and JSON bodies. Unknown metadata fields supplied by
the SDK are tolerated; numeric limits are validated.

| Route | Request fields | Response |
| --- | --- | --- |
| `/similar-tracks` | `song: {id}`, `count` | `{matches: [{song: SongRef, similarity}]}` |
| `/sonic-path` | `startSong: {id}`, `endSong: {id}`, `count` | Same envelope, ordered endpoints included; similarity `-1` |
| `/similar-artists` | `id`, `limit` | `{artists: [{id, name}]}` |
| `/artist-radio` | `id`, `count` | `{songs: [SongRef]}` |
| `/lyrics` | `track: {id}` | `{lyrics: [{lang, text}]}` |

`SongRef` contains Navidrome's source `id`, `name`, `artist`, `album` and
`durationMs`. Similar-songs-by-track maps the sonic response to the metadata
`{songs: [...]}` envelope inside the adapter. The Settings management API is
`/analysis/settings/integrations/navidrome` with GET/PUT, POST `/keys` and DELETE
`/keys/{id}`; it requires the normal Echora user session.

## Build and validation

### Plugin releases

The plugin has independent `X.Y.Z` versioning in `manifest.json`. Run the
**Release Navidrome plugin** GitHub Actions workflow from `main` and select
`patch`, `minor`, or `major`. For example, `0.1.0` becomes `0.1.1`, `0.2.0`, or
`1.0.0`, respectively. The action tests the adapter and Navidrome contracts,
builds the WASM module, and validates the `.ndp` archive before committing the
manifest version and pushing a `navidrome-plugin-vX.Y.Z` tag. It attaches
`echora-navidrome-X.Y.Z.ndp` and its `.sha256` checksum to a draft release, then
publishes it. Build artifacts are also retained on the workflow run.

This workflow does not change the Echora application version, build Docker
images, or replace the application's latest release. Generated files remain
ignored by Git. If publishing fails after the tag is pushed, the successful
build's workflow artifacts can be attached to that tag's release manually.

### Local builds

Go 1.25+ and TinyGo 0.42.0 are needed for the plugin; parser contract tests require
Go 1.27 because the pinned Navidrome source does. Python 3 is used for packaging.

```sh
make test
make package
cd contract-tests
go test ./...
```

The package contains exactly `manifest.json` and `plugin.wasm`. Generated packages
are ignored by Git. Backend tests in
`services/analysis/tests/test_navidrome_plugin.py` validate the exporter against
the checked-in parser fixtures and exercise the actual credential migration on
an isolated PostgreSQL schema when `TEST_DATABASE_URL` is set. Without that
variable, database tests skip; use a disposable test database, never production.

For environments with limited temporary storage, set `TMPDIR`, `GOCACHE` and
`GOMODCACHE` to directories on a sufficiently large volume before building.

Before production rollout, verify on the target Navidrome build: plugin loads and
shows SonicSimilarity/Metadata/Lyrics capabilities; sonic endpoints return valid
local IDs; metadata radio works; lyrics fall back for missing tracks; translations
and timing render in the intended clients; revoked/expired keys stop working.

References: [Navidrome plugins](https://www.navidrome.org/docs/usage/features/plugins/),
[AudioMuse integration](https://www.navidrome.org/docs/usage/integration/audiomuse/),
[pinned SDK/source](https://github.com/navidrome/navidrome/tree/0e1893530b844898cbf87825fdefd4b13e1ff3cc).
