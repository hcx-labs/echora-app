"""``PUT /library/tracks/{track_id}/lyrics``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from backend.library.db import require_visible
from core.db.connection import connect
from core.errors import InvalidQueryError

UPSERT_SQL = """INSERT INTO lyrics (track_id, source, text, language, provenance, availability_status)
   VALUES (%s,'manual',%s,%s,jsonb_build_object('manual',true,'edited_at',now()::text),'available')
   ON CONFLICT (track_id) DO UPDATE SET source='manual', text=EXCLUDED.text,
     language=EXCLUDED.language, provenance=(CASE WHEN lyrics.text IS DISTINCT FROM EXCLUDED.text THEN
         coalesce(lyrics.provenance,'{}'::jsonb) - 'lines' - 'synced'
       ELSE coalesce(lyrics.provenance,'{}'::jsonb) END)
       || jsonb_build_object('manual',true,'edited_at',now()::text),
     availability_status='available', created_at=now()"""


class LyricsUpdate(PutRequest):
    """Lyrics typed by the user, and optionally their language."""

    text: str = Field(min_length=1, max_length=100_000)
    language: str | None = Field(default=None, max_length=12)


class LyricsUpdated(ResponseModel):
    """Whether karaoke and the lyrics embedding are redone (only when the text changed)."""

    track_id: str
    karaoke_pending: bool
    lyrics_embedding_pending: bool


class TrackLyricsUpdate(Endpoint):
    """Replace a song's lyrics with the user's own."""

    method = "PUT"
    path = "/tracks/{track_id}/lyrics"
    request = LyricsUpdate
    response = LyricsUpdated

    def handle(
        self, track_id: uuid.UUID, body: LyricsUpdate, user: SessionUser = Depends(current_user)
    ) -> dict:
        lyric_text = body.text.strip()
        if not lyric_text:
            raise InvalidQueryError("Lyrics cannot be empty")
        language = body.language.strip().lower() if body.language else None
        with connect(dict_rows=True) as connection, connection.cursor() as cursor:
            require_visible(cursor, user["id"], track_id, "Track is not in your library")
            cursor.execute("SELECT text, language FROM lyrics WHERE track_id=%s", (track_id,))
            existing = cursor.fetchone()
            changed = existing is None or existing["text"] != lyric_text
            cursor.execute(UPSERT_SQL, (track_id, lyric_text, language))
            if changed:
                cursor.execute("DELETE FROM karaoke_lyrics_variants WHERE track_id=%s", (track_id,))
                cursor.execute(
                    "DELETE FROM embeddings WHERE track_id=%s AND embedding_type='lyrics'",
                    (track_id,),
                )
        return {
            "track_id": str(track_id),
            "karaoke_pending": changed,
            "lyrics_embedding_pending": changed,
        }
