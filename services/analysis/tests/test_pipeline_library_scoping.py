"""Dependency-free orchestration regressions; execute real function bodies with fake IO."""

import ast
from contextlib import nullcontext
import functools
import hashlib
import logging
from echora_analysis.settings import Settings
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
import uuid


SOURCE = Path(__file__).parents[1] / "src" / "echora_analysis"


def load_functions(filename, **dependencies):
    tree = ast.parse((SOURCE / filename).read_text())
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ] + [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    namespace = dict(
        get_settings=lambda: Settings.model_construct(database_url="fake"), **dependencies
    )
    exec(compile(ast.fix_missing_locations(tree), filename, "exec"), namespace)
    return namespace


class Cancelled(BaseException):
    pass


class LibraryScopingTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.cursor = self.connection.cursor.return_value.__enter__.return_value
        self.library_id = uuid.uuid4()

    def plan_namespace(self):
        return load_functions(
            "processing_plan.py",
            os=SimpleNamespace(environ={}),
            ProcessingPlan=lambda **kw: SimpleNamespace(**kw),
            feature_enabled=lambda connection, feature, where=None: True,
        )

    def test_planners_scope_optional_library_and_ids(self):
        ns = self.plan_namespace()
        self.cursor.fetchone.return_value = (True,)
        self.cursor.fetchall.return_value = [("shared-id",)]
        for name, args in [("plan_lyrics", ()), ("plan_karaoke", ("pipeline",))]:
            for ids in (None, [], ["shared-id"]):
                for library in (None, self.library_id):
                    with self.subTest(name=name, ids=ids, library=library):
                        ns[name](self.connection, *args, external_ids=ids, library_id=library)
                        sql, params = self.cursor.execute.call_args.args
                        self.assertEqual("ts.library_id=%s" in sql, library is not None)
                        self.assertEqual("ts.external_id=ANY(%s)" in sql, ids is not None)
                        if library is not None:
                            self.assertIn(library, params)
                        if ids is not None:
                            self.assertIn(ids, params)

    def test_karaoke_plans_any_lyric_text(self):
        # Aligned from the audio alone: plain (unsynced) lyrics qualify too.
        self.plan_namespace()["plan_karaoke"](self.connection, "pipeline")
        sql = self.cursor.execute.call_args.args[0]
        self.assertIn("btrim(l.text)", sql)
        self.assertNotIn("'synced'", sql)

    def test_library_resolution_fails_closed(self):
        resolve = self.plan_namespace()["resolve_library_id"]
        for rows in ([], [(uuid.uuid4(),), (uuid.uuid4(),)]):
            self.cursor.fetchall.return_value = rows
            with self.assertRaises(ValueError):
                resolve(self.connection, "https://music/")
        self.cursor.fetchall.return_value = [(self.library_id,)]
        self.assertEqual(resolve(self.connection, "https://music/"), self.library_id)
        sql, params = self.cursor.execute.call_args.args
        self.assertIn("lower(rtrim(root_path", sql)
        self.assertEqual(params, ("https://music/",))

    def pipeline_namespace(self, kind):
        client = MagicMock()
        connect = MagicMock()
        connect.return_value.__enter__.return_value = self.connection
        navidrome = MagicMock()
        navidrome.return_value.__enter__.return_value = client
        ns = load_functions(
            f"{kind}_pipeline.py",
            psycopg=SimpleNamespace(connect=connect),
            os=SimpleNamespace(environ={"DATABASE_URL": "fake"}),
            NavidromeClient=navidrome,
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            prepare_audio=MagicMock(),
            get_check=lambda: lambda: None,
            functools=functools,
            hashlib=hashlib,
            current_remote=lambda: None,
            separation_phase=nullcontext,
            SongTranscriber=MagicMock(),
            needs_repair=lambda draft: False,
            diagnostic_writer=MagicMock(),
            logging=logging,
        )
        for name in (
            "resolve_library_id",
            "configure_representations",
            "_create_run",
            "start_attempt",
            "record_track",
            "finish_attempt",
            "_store_lyrics",
            "store_original",
            "release_model",
            "_stop_fa_kara_worker",
            "_stored_model_revision",
        ):
            ns[name] = MagicMock()
        ns["resolve_library_id"].return_value = self.library_id
        return ns, client

    def test_lyrics_scope_and_cleanup_on_cancellation(self):
        ns, client = self.pipeline_namespace("lyrics")
        ns["plan_lyrics"] = MagicMock(
            return_value=SimpleNamespace(lyrics_external_ids=("shared-id",))
        )
        model = MagicMock()
        model.embed.side_effect = Cancelled()
        ns["LyricsEmbeddingModel"] = MagicMock(return_value=model)
        client.lyrics.return_value = {"text": "words", "status": "available"}
        self.cursor.fetchall.return_value = [(uuid.uuid4(), "shared-id", "Title")]
        with self.assertRaises(Cancelled):
            ns["backfill_lyrics"]("https://music", "user", "password")
        ns["plan_lyrics"].assert_called_once_with(self.connection, None, library_id=self.library_id)
        # The track selection; per-track lyric lookups follow it.
        sql, params = next(
            call.args
            for call in self.cursor.execute.call_args_list
            if "track_sources" in call.args[0]
        )
        self.assertIn("ts.library_id=%s", sql)
        self.assertEqual(params, (["shared-id"], self.library_id))
        ns["release_model"].assert_called_once_with(model)
        ns["finish_attempt"].assert_not_called()

    def test_lyrics_cleanup_when_run_setup_fails(self):
        ns, _ = self.pipeline_namespace("lyrics")
        ns["plan_lyrics"] = MagicMock(
            return_value=SimpleNamespace(lyrics_external_ids=("shared-id",))
        )
        model = MagicMock()
        ns["LyricsEmbeddingModel"] = MagicMock(return_value=model)
        ns["_create_run"].side_effect = RuntimeError("database unavailable")
        self.cursor.fetchall.return_value = [(uuid.uuid4(), "shared-id", "Title")]
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            ns["backfill_lyrics"]("https://music", "user", "password")
        ns["release_model"].assert_called_once_with(model)

    def test_pipelines_never_select_sources_when_library_resolution_fails(self):
        for kind in ("lyrics", "karaoke", "voice"):
            with self.subTest(kind=kind):
                ns, client = self.pipeline_namespace(kind)
                ns["resolve_library_id"].side_effect = ValueError("unknown library")
                ns["_KARAOKE_LOCK"] = MagicMock()
                with self.assertRaisesRegex(ValueError, "unknown library"):
                    ns[f"backfill_{kind}"]("https://unknown", "user", "password")
                client.audio_bytes.assert_not_called()
                client.lyrics.assert_not_called()
                ns["configure_representations"].assert_not_called()
                self.cursor.execute.assert_not_called()

    def test_karaoke_scope_and_cleanup_on_cancellation(self):
        ns, client = self.pipeline_namespace("karaoke")
        ns.update(
            _KARAOKE_LOCK=MagicMock(),
            DEFAULT_MODEL_REVISION="model",
            KARAOKE_PIPELINE_REVISION="pipeline",
        )
        ns["plan_karaoke"] = MagicMock(
            return_value=SimpleNamespace(karaoke_external_ids=("shared-id",))
        )
        client.audio_bytes.side_effect = Cancelled()
        self.cursor.fetchall.return_value = [
            (uuid.uuid4(), "shared-id", "Title", "words", "en", [])
        ]
        with self.assertRaises(Cancelled):
            ns["backfill_karaoke"]("https://music", "user", "password")
        self.assertEqual(ns["plan_karaoke"].call_args.kwargs, {"library_id": self.library_id})
        sql, params = self.cursor.execute.call_args.args
        self.assertIn("ts.library_id=%s", sql)
        self.assertEqual(params, (["shared-id"], self.library_id))
        self.assertEqual(ns["_stop_fa_kara_worker"].call_count, 2)  # Before prep and finally.

    def test_voice_scope_ids_and_cleanup_on_cancellation(self):
        for ids in (None, [], ["shared-id"]):
            with self.subTest(ids=ids):
                ns, client = self.pipeline_namespace("voice")
                model = MagicMock()
                ns.update(
                    _id_filter=self.plan_namespace()["_id_filter"],
                    VOICE_EMBEDDING_TYPE="voice-gender",
                    _model=(model,),
                    _model_lock=MagicMock(),
                )
                ns["shared_voice_model"] = MagicMock(return_value=model)
                ns["_fetch_stream"] = MagicMock(side_effect=Cancelled())
                self.cursor.fetchall.return_value = [(uuid.uuid4(), "shared-id", "Title")]
                with self.assertRaises(Cancelled):
                    ns["backfill_voice"]("https://music", "user", "password", external_ids=ids)
                sql, params = self.cursor.execute.call_args.args
                self.assertIn("ts.library_id=%s", sql)
                self.assertEqual(params[0], self.library_id)
                self.assertEqual("ts.external_id=ANY(%s)" in sql, ids is not None)
                if ids is not None:
                    self.assertEqual(params[1], ids)
                self.assertIsNone(ns["_model"])
                self.connection.rollback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
