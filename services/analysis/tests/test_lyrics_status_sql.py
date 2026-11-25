"""Run the endpoint's actual upserts against PostgreSQL, including type inference."""

import ast
import os
import unittest
from pathlib import Path
from uuid import uuid4

import psycopg


def endpoint_upsert(name):
    source = Path(__file__).resolve().parents[1] / "src/echora_analysis/main.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "INSERT INTO lyrics " in node.value
    )


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "TEST_DATABASE_URL required")
class LyricsUpsertTests(unittest.TestCase):
    def setUp(self):
        self.db = psycopg.connect(os.environ["TEST_DATABASE_URL"])
        self.addCleanup(self.db.close)
        self.db.execute("""CREATE TEMP TABLE lyrics (
            track_id uuid PRIMARY KEY, source text NOT NULL, text text,
            language text, provenance jsonb, availability_status text NOT NULL,
            created_at timestamptz DEFAULT now())""")

    def test_instrumental_and_missing_insert_and_update(self):
        query = endpoint_upsert("update_track_lyrics_status")
        for first, second in [("instrumental", "missing"), ("missing", "instrumental")]:
            track = uuid4()
            for status in [first, second]:
                self.db.execute(query, (track, status, status))
                row = self.db.execute(
                    "SELECT availability_status,provenance FROM lyrics WHERE track_id=%s", (track,)
                ).fetchone()
                self.assertEqual(row[0], status)
                self.assertEqual(row[1]["manual_status"], status)

    def test_forced_transcription_insert_and_update(self):
        query = endpoint_upsert("force_transcription_language")
        track = uuid4()
        for language in ["en", "ja"]:
            self.db.execute(query, (track, language, language))
            row = self.db.execute(
                "SELECT availability_status,provenance FROM lyrics WHERE track_id=%s", (track,)
            ).fetchone()
            self.assertEqual(row[0], "missing")
            self.assertEqual(row[1]["transcription_language"], language)
            self.assertTrue(row[1]["forced_transcription"])


if __name__ == "__main__":
    unittest.main()
