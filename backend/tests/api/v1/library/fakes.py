"""A fake psycopg connection: queued fetchone/fetchall results, recorded executes."""

from __future__ import annotations

from types import SimpleNamespace


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.executed: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        return self.rows.pop(0) if self.rows else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection(FakeCursor):
    def cursor(self, **kwargs):
        return self


def install(monkeypatch, module, rows, name="connect"):
    """Make ``module.<name>()`` return one fake connection; returns it."""
    connection = FakeConnection(rows)
    monkeypatch.setattr(module, name, lambda *a, **k: connection)
    return connection


__all__ = ["FakeConnection", "SimpleNamespace", "install"]
