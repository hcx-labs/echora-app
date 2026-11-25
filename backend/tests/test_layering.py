"""The backend reaches the database only through core stores (``core/<domain>/store.py``)."""

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"

# SQL, raw connections and ORM queries or writes. Endpoints may still read and set attributes of
# the ORM objects a store returns; the store's caller owns the transaction.
FORBIDDEN = re.compile(
    r"\.execute(many)?\(|\.cursor\(|^\s*(from|import) psycopg"
    r"|^\s*from sqlalchemy import|\bsession\.(scalar|scalars|get|add|delete|flush|query)\("
    r"|\b(SELECT|INSERT INTO|UPDATE \w+ SET|DELETE FROM)\b",
    re.MULTILINE,
)


def test_backend_has_no_queries():
    offenders = [
        f"{path.relative_to(SOURCE)}: {match.group(0).strip()}"
        for path in sorted(SOURCE.rglob("*.py"))
        for match in FORBIDDEN.finditer(path.read_text())
    ]
    assert offenders == []
