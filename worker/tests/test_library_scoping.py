"""Dependency-free orchestration regressions; execute real function bodies with fake IO."""

import ast
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.config import config_classes
from core.planning import store as planning_store

SOURCE = Path(__file__).parents[1] / "src"


def load_functions(filename, **dependencies):
    tree = ast.parse((SOURCE / filename).read_text())
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ] + [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    namespace = {cls.__name__: cls for cls in config_classes()}
    namespace.update(
        DatabaseConfig=SimpleNamespace(load=lambda: SimpleNamespace(database_url="fake")),
        **dependencies,
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
            "planning/audio.py",
            os=SimpleNamespace(environ={}),
            ProcessingPlan=lambda **kw: SimpleNamespace(**kw),
            feature_enabled=lambda connection, feature, where=None: True,
            store=planning_store,
        )

    def test_planners_scope_optional_library_and_ids(self):
        ns = self.plan_namespace()
        self.cursor.fetchone.return_value = (True,)
        self.cursor.fetchall.return_value = [{"external_id": "shared-id"}]
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
        for rows in ([], [{"id": uuid.uuid4()}, {"id": uuid.uuid4()}]):
            self.cursor.fetchall.return_value = rows
            with self.assertRaises(ValueError):
                resolve(self.connection, "https://music/")
        self.cursor.fetchall.return_value = [{"id": self.library_id}]
        self.assertEqual(resolve(self.connection, "https://music/"), self.library_id)
        sql, params = self.cursor.execute.call_args.args
        self.assertIn("lower(rtrim(root_path", sql)
        self.assertEqual(params, ("https://music/",))


if __name__ == "__main__":
    unittest.main()
