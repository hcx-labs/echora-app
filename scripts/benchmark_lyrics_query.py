"""Ad-hoc benchmark: embed free-text queries and rank lyrics embeddings by cosine similarity."""

from __future__ import annotations

import os
import sys
import time
import uuid

import numpy as np
import psycopg

from core.embeddings.text import LyricsEmbeddingModel

QUERIES = [
    "relationship breakup broken heart",
    "dancing all night at the club",
    "feeling hopeful about the future",
    "angry defiant rebellion",
]


def _run_queries(model: LyricsEmbeddingModel, queries: list[str], top_k: int = 10) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT DISTINCT ON (e.track_id) e.track_id, t.title, t.artist, e.embedding
                   FROM embeddings e
                   JOIN analysis_runs r ON r.id = e.run_id
                   JOIN tracks t ON t.id = e.track_id
                   WHERE e.embedding_type='lyrics' AND e.aggregation='normalized_mean'
                   AND r.model_name='bge_m3'
                   ORDER BY e.track_id, e.id DESC""",
            )
            rows = cursor.fetchall()
    print(f"Loaded {len(rows)} track-level lyrics embeddings\n")

    def parse(value):
        if isinstance(value, str):
            return np.fromstring(value.strip("[]"), sep=",", dtype=np.float32)
        return np.array(value, dtype=np.float32)

    matrix = np.array([parse(row[3]) for row in rows])
    track_ids = [row[0] for row in rows]

    for query in queries:
        started = time.perf_counter()
        result = model.embed(query)
        elapsed_ms = (time.perf_counter() - started) * 1000
        query_vec = np.array(result.aggregate, dtype=np.float32)
        sims = matrix @ query_vec
        order = np.argsort(-sims)[:top_k]
        print(f"query: {query!r}  (embed {elapsed_ms:.0f}ms)")
        print(
            f"  top similarity: {sims[order[0]]:.4f}   median: {np.median(sims):.4f}   p95: {np.percentile(sims, 95):.4f}"
        )
        for rank, idx in enumerate(order, 1):
            row = rows[idx]
            print(f"  {rank:2d}. {sims[idx]:.4f}  {row[1]} — {row[2] or '?'}")
        print()


if __name__ == "__main__":
    queries = sys.argv[1:] or QUERIES
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = LyricsEmbeddingModel(
        os.environ.get("LYRICS_MODEL_ID", "BAAI/bge-m3"),
        os.environ.get("LYRICS_REVISION", "5617a9f61b028005a4858fdac845db406aefb181"),
        device,
    )
    try:
        _run_queries(model, queries)
    finally:
        from core.embeddings.audio_models import release_model

        release_model(model)
