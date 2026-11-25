"""External processing: run analysis model computation on Modal (see docs/modal-compute.md).

Instance-wide and admin-managed, like External AI. Admins store a Modal token
(the secret encrypted with CREDENTIAL_ENCRYPTION_KEY) and prepare the
workspace, which deploys Echora's analysis app and downloads its models there.
Every Modal batch re-checks that deployment under a database lock, so an Echora
upgrade or a model change is deployed or downloaded before any song is sent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

from core.db.config import require_database_url
from worker.remote.compute import compute_settings
from worker.remote.config import RemoteConfig
from worker.remote.constants import GPU_TYPES
from worker.remote.session import ModalConfig, ModalSession

logger = logging.getLogger(__name__)

# Serializes deployment and model preparation across workers and batches.
PREPARE_LOCK = 0x45434D4F44414C  # "ECMODAL"
SETUP_JOB = "modal_setup"


def _connect():
    return psycopg.connect(require_database_url(), row_factory=dict_row)


def load(db=None) -> dict[str, object]:
    """The stored settings row, or defaults when none was saved."""
    if db is None:
        with _connect() as db:
            return load(db)
    row = db.execute("SELECT * FROM external_processing_settings WHERE singleton").fetchone()
    return (
        dict(row)
        if row
        else {
            "enabled": False,
            "provider": "modal",
            "token_id": "",
            "token_secret_encrypted": None,
            "gpu": "L40S",
            "default_compute": "local",
            "allow_users": False,
            "workspace": None,
            "status": "unprepared",
            "status_detail": None,
            "deployed_code": None,
            "deployed_models": None,
            "checked_at": None,
        }
    )


def _cipher():
    from core.library.connections import credential_cipher as cipher

    return cipher()


def config(row: dict[str, object] | None = None) -> ModalConfig | None:
    """Credentials and deployment target, when Modal is enabled and fully configured."""
    row = row or load()
    if not row["enabled"] or not row["token_id"] or not row["token_secret_encrypted"]:
        return None
    secret = _cipher().decrypt(bytes(row["token_secret_encrypted"])).decode()
    from worker.models.huggingface_token import token_or_none

    # Gated models (LTX-2.5) download with the shared token from Analysis models settings.
    hf_token = token_or_none()
    return ModalConfig(
        str(row["token_id"]), secret, str(row["gpu"]), RemoteConfig.load().image, hf_token
    )


def remote_settings() -> dict[str, object]:
    """The settings Modal computes with.

    The worker's model settings, and whether Modal's storage holds motion artwork models (its
    switch for syncs on Modal).
    """
    from worker.artwork.settings import _connect as motion_connect, load_settings as motion_settings

    with motion_connect() as connection, connection.cursor() as cursor:
        artwork = motion_settings(cursor)
    return compute_settings(motion_artwork_models=artwork.enabled and artwork.generate_on_modal)


def _is_admin(db, user_id) -> bool:
    row = db.execute("SELECT is_admin FROM users WHERE id=%s", (user_id,)).fetchone()
    return bool(row and row["is_admin"])


def compute_for(user_id, requested: str | None) -> str:
    """Where a new job's model computation runs: the request, or the instance default.

    Raises PermissionError when the user asked for Modal but may not use it.
    """
    if requested == "local":
        return "local"
    with _connect() as db:
        row = load(db)
        usable = bool(row["enabled"] and row["token_id"] and row["token_secret_encrypted"])
        allowed = usable and (row["allow_users"] or _is_admin(db, user_id))
    if requested == "modal":
        if not allowed:
            raise PermissionError("Modal processing is not available")
        return "modal"
    return "modal" if allowed and row["default_compute"] == "modal" else "local"


def _record(db, **values) -> None:
    db.execute(
        "INSERT INTO external_processing_settings (singleton) VALUES (true) ON CONFLICT DO NOTHING"
    )
    assignments = ", ".join(f"{name}=%s" for name in values)
    db.execute(
        f"UPDATE external_processing_settings SET {assignments} WHERE singleton",
        tuple(values.values()),
    )


def prepare(
    progress: Callable[[dict[str, object]], None] = lambda _: None,
    check: Callable[[], None] = lambda: None,
    *,
    session_factory=ModalSession,
) -> ModalSession:
    """Bring the Modal workspace up to date with this worker, and return a ready session."""
    with _connect() as db:
        db.autocommit = True
        progress({"phase": "modal", "message": "Waiting for other Modal preparation"})
        while not db.execute(
            "SELECT pg_try_advisory_lock(%s) AS locked", (PREPARE_LOCK,)
        ).fetchone()["locked"]:
            check()
            import time

            time.sleep(2)
        try:
            settings = config()
            if settings is None:
                raise RuntimeError("Modal processing is not enabled and configured")
            _record(db, status="preparing", status_detail=None)
            try:
                session = session_factory(settings, remote_settings())
                outcome = session.ensure_ready(progress, check)
            except Exception as error:
                detail = (
                    str(error)
                    if isinstance(error, RuntimeError)
                    else "Modal preparation failed; see service logs"
                )
                logger.error("Modal preparation failed: %s", type(error).__name__)
                _record(
                    db,
                    status="failed",
                    status_detail=detail[-1500:],
                    checked_at=datetime.now().astimezone(),
                )
                raise
            _record(
                db,
                status="ready",
                status_detail=None,
                deployed_code=outcome["code"],
                deployed_models=outcome["models"],
                checked_at=datetime.now().astimezone(),
            )
            return session
        finally:
            db.execute("SELECT pg_advisory_unlock(%s)", (PREPARE_LOCK,))


def open_session(
    progress: Callable[[dict[str, object]], None] = lambda _: None,
    check: Callable[[], None] = lambda: None,
) -> ModalSession:
    """A batch's Modal session, checked against this worker before any upload."""
    return prepare(progress, check)


def execute_setup(job: dict, context) -> dict[str, object]:
    """The "Prepare Modal" job: deploy and download whatever the workspace lacks."""

    def report(update):
        context.check()
        context.report(update)

    session = prepare(report, context.check)
    session.close()
    report({"phase": "complete", "message": "Modal is ready"})
    return {"modal": "ready"}


def _public(
    row: dict[str, object], setup: dict[str, object] | None, has_hf_token: bool
) -> dict[str, object]:
    checked = row.get("checked_at")
    return {
        "enabled": row["enabled"],
        "provider": row["provider"],
        "token_id": row["token_id"],
        "has_secret": bool(row["token_secret_encrypted"]),
        "gpu": row["gpu"],
        "has_hf_token": has_hf_token,
        "default_compute": row["default_compute"],
        "allow_users": row["allow_users"],
        "workspace": row["workspace"],
        "status": row["status"],
        "status_detail": row["status_detail"],
        "checked_at": checked.isoformat() if isinstance(checked, datetime) else None,
        "image": RemoteConfig.load().image,
        "gpu_types": list(GPU_TYPES),
        "setup": setup,
    }


def _has_hf_token(db) -> bool:
    """Whether a Hugging Face token is stored; presence only, the token is not decrypted."""
    row = db.execute(
        "SELECT hf_token_encrypted IS NOT NULL AS stored FROM analysis_settings WHERE singleton"
    ).fetchone()
    return bool(row and row["stored"])


def _setup_job(db) -> dict[str, object] | None:
    row = db.execute(
        """SELECT id, status, progress, error, created_at FROM jobs WHERE kind=%s
                        ORDER BY created_at DESC LIMIT 1""",
        (SETUP_JOB,),
    ).fetchone()
    if not row:
        return None
    progress = row["progress"] or {}
    return {
        "id": str(row["id"]),
        "status": row["status"],
        "error": row["error"],
        **{key: progress.get(key) for key in ("phase", "message", "completed", "total")},
    }
