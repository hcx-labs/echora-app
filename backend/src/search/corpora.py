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
from core.embeddings import store
from core.errors import ConflictError


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
            return store.corpus_digest(connection, self.kind, self.scope)

    def open(self) -> None:
        with connect() as connection:
            rows = store.user_corpus(connection, self.kind, self.scope)
        self.corpus: Corpus | None = None
        if rows:
            matrix = np.stack([store.parse_vector(row.pop("embedding")) for row in rows])
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
