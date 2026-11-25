"""core and worker hold no HTTP code, and core never depends on the worker."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


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
        ("core/src", ("fastapi", "starlette", "worker", "echora_analysis", "backend")),
        ("worker/src", ("fastapi", "starlette", "echora_analysis", "backend")),
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
