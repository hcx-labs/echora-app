"""Layering: no HTTP code in core or the worker, no worker in core, no worker queries.

The worker reaches the database only through core stores (``core/<domain>/store.py``).
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "worker/src"


def imported_modules(package: Path):
    for path in package.rglob("*.py"):
        if "vendor" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                yield path, [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                yield path, [node.module]


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("core/src", ("fastapi", "starlette", "worker", "backend")),
        ("worker/src", ("fastapi", "starlette", "backend")),
    ],
)
def test_no_forbidden_imports(package, forbidden):
    offenders = sorted(
        f"{path.relative_to(ROOT)}: {name}"
        for path, names in imported_modules(ROOT / package)
        for name in names
        if name.split(".")[0] in forbidden
    )
    assert offenders == []


# The same patterns as backend/tests/test_layering.py: SQL, raw connections and ORM queries.
FORBIDDEN = re.compile(
    r"\.execute(many)?\(|\.cursor\(|^\s*(from|import) psycopg"
    r"|^\s*from sqlalchemy import|\bsession\.(scalar|scalars|get|add|delete|flush|query)\("
    r"|\b(SELECT|INSERT INTO|UPDATE \w+ SET|DELETE FROM)\b",
    re.MULTILINE,
)

# Stages hold the connection they pass to the stores; these modules import psycopg only to
# annotate it (``psycopg.Connection``).
CONNECTION_ANNOTATIONS = {
    "fusion/pipeline.py",
    "library/fingerprints.py",
    "library/ingest.py",
    "library/recordings.py",
    "lyrics/pipeline.py",
    "melody/corpus.py",
    "melody/index.py",
    "planning/audio.py",
    "profiles/pipeline.py",
    "voice/pipeline.py",
}
ALLOWED = {(path, "import psycopg") for path in CONNECTION_ANNOTATIONS} | {
    # A model download strategy's ``strategy.execute()``, not a query.
    ("models/download.py", ".execute("),
}


def test_worker_has_no_queries():
    offenders = [
        (str(path.relative_to(WORKER)), match.group(0).strip())
        for path in sorted(WORKER.rglob("*.py"))
        for match in FORBIDDEN.finditer(path.read_text())
    ]
    assert [offender for offender in offenders if offender not in ALLOWED] == []


def test_the_allowlist_has_no_stale_entries():
    found = {
        (str(path.relative_to(WORKER)), match.group(0).strip())
        for path in sorted(WORKER.rglob("*.py"))
        for match in FORBIDDEN.finditer(path.read_text())
    }
    assert ALLOWED - found == set()
