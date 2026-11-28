# Processing information and resyncs

Every song's derived data carries the revision that made it: karaoke alignments their pipeline
and model revision, stems their recipe, translations their model and prompt, themes their
prompt revision, AI lyrics their transcription pipeline. **Processing information** (the
three-dots menu of a song in Your library) shows each one against what the current code would
make: up to date, out of date, not made, or failed.

## Resync a song

Each pipeline can be marked for resync there (`PUT /api/v1/library/tracks/{id}/processing/
{pipeline}`; `DELETE` withdraws it). Nothing runs at once: the request is a row in
`track_resync_requests`, and the next sync, **partial or full**, takes the song
(`worker/planning/sync.py`) and redoes that pipeline whatever its stored revision. The stage
that redoes it removes the request.

| pipeline | what the next sync does |
| --- | --- |
| `transcription` | replaces the lyrics with an AI transcription in the chosen language (needs AI lyrics on) |
| `karaoke` | realigns; from the song's stored model reading when it has a current one (fast) |
| `karaoke_stems` | separates and encodes the stems again, also after a failure |
| `translation` | translates again into every configured target, without reusing ready ones |
| `themes` | describes the lyrics' themes again |

Embeddings are shown for information only: lyrics embeddings follow the lyrics, audio
embeddings are kept current by every full sync.

## Full syncs and karaoke revisions

Settings → Analysis models → **Realign karaoke on full sync** (`analysis_settings.
karaoke_realign_outdated`, on by default). On, a full sync realigns every song whose alignment
is of another pipeline (`KARAOKE_PIPELINE_REVISION`) or model revision. Off, full syncs only
align songs without karaoke (and songs whose lyrics changed, which discards their alignment);
songs move to a new revision one at a time, from Processing information. An administrator's
karaoke backfill always realigns.
