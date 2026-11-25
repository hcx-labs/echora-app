"""Durable execution status, separate from reusable representation definitions."""

from __future__ import annotations

import os

from core.analysis import store


def start_attempt(connection, run_id, requested: int):
    return store.start_attempt(connection, run_id, requested, os.getenv("ECHORA_JOB_ID"))


def record_track(connection, attempt_id, external_id: str, track_id=None, error: str | None = None):
    store.record_attempt_track(connection, attempt_id, external_id, track_id, error)


def finish_attempt(connection, attempt_id):
    store.finish_attempt(connection, attempt_id)
