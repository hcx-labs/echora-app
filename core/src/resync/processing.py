"""What each pipeline stored for one song: the rows behind "Processing information"."""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection

from core.db.connection import fetch_all, fetch_one


def track_outputs(db: Connection, track_id: uuid.UUID) -> dict[str, Any]:
    """The song's lyrics, karaoke, stems, translations, themes and embedding runs."""
    return {
        "lyrics": fetch_one(
            db,
            "SELECT source, language, provenance, created_at FROM lyrics WHERE track_id=%s",
            (track_id,),
        ),
        "karaoke": fetch_one(
            db,
            """SELECT model, model_revision,
                      provenance->>'pipeline_revision' AS pipeline_revision, created_at
               FROM karaoke_lyrics_variants WHERE track_id=%s AND NOT bounded""",
            (track_id,),
        ),
        "stems": fetch_one(
            db,
            "SELECT recipe, status, created_at FROM karaoke_stems WHERE track_id=%s",
            (track_id,),
        ),
        "translations": fetch_all(
            db,
            """SELECT source_language, target_language, status, model, prompt_revision,
                      updated_at
               FROM lyric_translations WHERE track_id=%s AND provenance='ai'
               ORDER BY target_language, source_language""",
            (track_id,),
        ),
        "themes": fetch_one(
            db,
            """SELECT prompt_revision, model, status, updated_at
               FROM track_lyric_themes WHERE track_id=%s""",
            (track_id,),
        ),
        "embedding_runs": {
            row["model_name"]: row["model_revision"]
            for row in fetch_all(
                db,
                """SELECT DISTINCT r.model_name, r.model_revision FROM current_embeddings e
                   JOIN analysis_runs r ON r.id=e.run_id WHERE e.track_id=%s""",
                (track_id,),
            )
        },
    }
