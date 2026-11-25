"""Compare seed-song neighborhoods: lyrics-only vs semantic fusion (lyrics+audio)."""

from __future__ import annotations

import os
import sys
import uuid

import numpy as np
import psycopg


def _matrix(cursor, embedding_type: str, model_filter: str | None = None):
    query = """SELECT DISTINCT ON (e.track_id) e.track_id, t.title, t.artist, e.embedding
               FROM current_embeddings e JOIN tracks t ON t.id=e.track_id
               WHERE e.embedding_type=%s AND e.window_index IS NULL"""
    params = [embedding_type]
    if model_filter:
        query += (
            " AND EXISTS (SELECT 1 FROM analysis_runs r WHERE r.id=e.run_id AND r.model_name=%s)"
        )
        params.append(model_filter)
    query += " ORDER BY e.track_id, e.id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    def parse(value):
        if isinstance(value, str):
            return np.fromstring(value.strip("[]"), sep=",", dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    return rows, np.array([parse(row[3]) for row in rows])


def _top(matrix, rows, seed_index: int, k: int = 10):
    sims = matrix @ matrix[seed_index]
    order = np.argsort(-sims)
    out = [(float(sims[i]), rows[i][1], rows[i][2]) for i in order[1 : k + 1]]
    return out


def main(seed_titles: list[str]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection, connection.cursor() as cursor:
        lyrics_rows, lyrics_matrix = _matrix(cursor, "lyrics", "bge_m3")
        fused_rows, fused_matrix = _matrix(cursor, "semantic_fusion")
        by_title = {row[1].lower(): i for i, row in enumerate(lyrics_rows)}
        fused_by_track = {row[0]: i for i, row in enumerate(fused_rows)}
        for title in seed_titles:
            index = by_title.get(title.lower())
            if index is None:
                print(f"seed not found in lyrics set: {title!r}")
                continue
            seed_track = lyrics_rows[index][0]
            fused_index = fused_by_track.get(seed_track)
            print(f"seed: {lyrics_rows[index][1]} — {lyrics_rows[index][2]}")
            if fused_index is None:
                print("  (no fused vector for this track)")
            for label, top in (
                ("lyrics-only", _top(lyrics_matrix, lyrics_rows, index)),
                (
                    "fused     ",
                    _top(fused_matrix, fused_rows, fused_index) if fused_index is not None else [],
                ),
            ):
                print(f"  {label}:")
                for score, t, artist in top:
                    print(f"    {score:.4f}  {t} — {artist}")
            print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["Malibu Nights", "Chandelier", "Look at the Sky"])
