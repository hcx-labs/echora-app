"""Per-user embedding corpora (MuQ-MuLan audio, BGE-M3 lyrics), kept in memory between requests.

Concept search, concept previews and the concept lens score every song of the user against a
text embedding, so each needs the user's whole matrix. A corpus stays open until a song's
current embedding is added, replaced or unlinked (``index_key``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from backend.search.service import SearchService, search_now
from core.db.connection import connect
from core.errors import ConflictError

_FILTERS = {
    "semantic": "e.embedding_type='audio-track' AND ar.model_name='muq_mulan'",
    "lyrics": "e.embedding_type='lyrics' AND e.window_index IS NULL AND ar.model_name='bge_m3'",
}


@dataclass(frozen=True)
class Corpus:
    """Songs (id, title, artist, album), their embeddings (one row each) and the run id."""

    rows: list[dict[str, object]]
    matrix: np.ndarray
    run_id: str


class _CorpusService(SearchService[None, Corpus]):
    kind: ClassVar[str]
    empty_message: ClassVar[str]
    persistent = True
    max_instances = 8

    scope: uuid.UUID

    def index_key(self) -> str:
        """A digest of every current (song, run) pair.

        Any added, replaced or unlinked embedding, or a switch of the active representation,
        changes it.
        """
        with connect() as connection:
            (digest,) = connection.execute(
                f"""SELECT coalesce(md5(string_agg(e.track_id::text || ':' || e.run_id::text, ','
                                                ORDER BY e.track_id, e.run_id)), '')
                    FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                    WHERE {_FILTERS[self.kind]}
                      AND EXISTS (SELECT 1 FROM user_track_links utl
                                  WHERE utl.track_id=e.track_id AND utl.user_id=%s)""",
                (self.scope,),
            ).fetchone()
        return str(digest)

    def open(self) -> None:
        with (
            connect(dict_rows=True) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                f"""SELECT DISTINCT ON (e.track_id) t.id, t.title, t.artist, t.album,
                           e.embedding::text AS embedding, ar.id AS run_id
                    FROM current_embeddings e
                    JOIN analysis_runs ar ON ar.id=e.run_id
                    JOIN tracks t ON t.id=e.track_id
                    WHERE {_FILTERS[self.kind]}
                      AND EXISTS (SELECT 1 FROM user_track_links utl
                                  WHERE utl.track_id=t.id AND utl.user_id=%s)
                    ORDER BY e.track_id, ar.created_at DESC""",
                (self.scope,),
            )
            rows = cursor.fetchall()
        self.corpus: Corpus | None = None
        if rows:
            matrix = np.stack(
                [np.fromstring(row.pop("embedding").strip("[]"), sep=",") for row in rows]
            )
            matrix.flags.writeable = False
            run_id = str(rows[0]["run_id"])
            for row in rows:
                row.pop("run_id")
            self.corpus = Corpus(rows, matrix, run_id)

    def search(self, query: None = None) -> Corpus:
        if self.corpus is None:
            raise ConflictError(self.empty_message)
        # Callers may annotate rows; the cached ones stay untouched.
        return Corpus(
            [dict(row) for row in self.corpus.rows], self.corpus.matrix, self.corpus.run_id
        )

    def close(self) -> None:
        self.corpus = None


class SemanticCorpus(_CorpusService):
    """The user's MuQ-MuLan audio embeddings."""

    name = "semantic-corpus"
    kind = "semantic"
    empty_message = "No MuQ-MuLan embeddings are available"


class LyricsCorpus(_CorpusService):
    """The user's BGE-M3 lyrics embeddings."""

    name = "lyrics-corpus"
    kind = "lyrics"
    empty_message = "No lyrics embeddings are available"


def semantic_corpus(user_id: uuid.UUID) -> tuple[list[dict[str, object]], np.ndarray, str]:
    """The user's MuQ-MuLan corpus (409 when the user has none)."""
    corpus = search_now(SemanticCorpus, user_id, None)
    return corpus.rows, corpus.matrix, corpus.run_id


def lyrics_corpus(user_id: uuid.UUID) -> tuple[list[dict[str, object]], np.ndarray, str]:
    """The user's BGE-M3 lyrics corpus (409 when the user has none)."""
    corpus = search_now(LyricsCorpus, user_id, None)
    return corpus.rows, corpus.matrix, corpus.run_id
