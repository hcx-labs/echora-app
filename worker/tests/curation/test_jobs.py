"""No live services: transactional fake DB and a fault-injectable remote client."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4
import ast
from pathlib import Path
import sys

import pytest
from core.errors import ConflictError, NotFoundError

import worker.curation.jobs as cj


class Result:
    def __init__(self, rows=()):
        self.rows = deepcopy(list(rows))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class Database:
    def __init__(self):
        self.curation = dict(id=uuid4(), user_id=uuid4(), navidrome_connection_id=uuid4(),
                             navidrome_playlist_id=None, status="draft", due=True)
        self.publication = None
        self.revisions = []
        self.events = []
        self.fail_revision = False
        self.acquired = True
        self.lock_held = False
        self.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.rollback() if args[0] else self.commit()

    def commit(self):
        self.saved = deepcopy((self.curation, self.publication, self.revisions))
        self.events.append("commit")

    def rollback(self):
        self.curation, self.publication, self.revisions = deepcopy(self.saved)
        self.events.append("rollback")

    def execute(self, sql, args=()):
        sql = " ".join(sql.split())
        self.events.append(sql)
        if "pg_try_advisory_lock" in sql:
            self.lock_held = self.acquired
            return Result([{"acquired": self.acquired}])
        if "pg_advisory" in sql:
            self.lock_held = "unlock" not in sql
            return Result([{"locked": True}])
        if sql.startswith("SELECT id FROM curations"):
            return Result([{"id": self.curation["id"]}] if self.curation and self.curation["due"] else [])
        if sql.startswith("SELECT * FROM curations"):
            found = self.curation and self.curation["id"] == args[0]
            if "AND user_id" in sql:
                found = found and self.curation["user_id"] == args[1]
            else:
                found = found and self.curation["due"]
            return Result([self.curation] if found else [])
        if sql.startswith("SELECT * FROM curation_publications"):
            p = self.publication
            found = p and (p["state"] != "complete" or (len(args) > 1 and p["job_id"] == args[1]))
            return Result([p] if found else [])
        if sql.startswith("INSERT INTO curation_publications"):
            self.publication = dict(id=args[0], curation_id=args[1], job_id=args[2],
                                    state="prepared", plan=deepcopy(args[3].obj), playlist_id=args[4])
            return Result([self.publication])
        if sql.startswith("UPDATE curation_publications"):
            if "state='publishing'" in sql:
                self.publication["state"] = "publishing"
            elif "state='complete'" in sql:
                self.publication["state"] = "complete"
            elif "job_id=%s" in sql:
                self.publication["job_id"] = args[0]
            else:
                self.publication["playlist_id"] = args[0]
            return Result()
        if sql.startswith("UPDATE curations"):
            if "status='failed'" in sql:
                self.curation.update(status="failed", last_error=args[0])
            elif "status='ready'" in sql:
                self.curation.update(status="ready")
            elif "navidrome_playlist_id=%s" in sql:
                self.curation["navidrome_playlist_id"] = args[0]
            else:
                self.curation.update(status="pending", due=False)
            return Result()
        if sql.startswith("INSERT INTO curation_revisions"):
            if self.fail_revision:
                raise RuntimeError("local revision write failed")
            self.revisions.append(args)
            return Result([{"id": uuid4()}])
        if sql.startswith("INSERT INTO curation_revision_tracks"):
            return Result()
        raise AssertionError((sql, args))


@pytest.fixture
def env(monkeypatch, library_helpers):
    db = Database()
    remote = SimpleNamespace(calls=[], error=None)
    plan = {"name": "Mix", "source_ids": ["b", "a", "b"], "recipe": {"shuffle_seed": 123},
            "revision_number": 1, "tracks": [dict(id=str(uuid4()), source_id=s, score=1,
             percentile=90, retained=False) for s in ["b", "a", "b"]]}
    selections = []

    def prepare(*args):
        selections.append(True)
        return deepcopy(plan)

    class Client:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def replace_playlist(self, name, ids, playlist_id):
            # Intent and exact ordered IDs must already be committed remotely.
            assert db.lock_held, "Serialization lock must span remote I/O"
            saved = db.saved[1]
            assert saved["state"] == "publishing"
            assert saved["plan"]["source_ids"] == ids
            assert saved["plan"]["recipe"] == {"shuffle_seed": 123}
            remote.calls.append((name, ids, playlist_id))
            if remote.error:
                raise remote.error
            return playlist_id or "created-id"

    main = SimpleNamespace(_load_connection=lambda _: ("url", "user", "secret"), NavidromeClient=Client)
    library_helpers(main)
    monkeypatch.setattr(cj, "_connect", lambda: db)
    monkeypatch.setattr(cj, "_prepare", prepare)
    job = {"id": uuid4(), "user_id": db.curation["user_id"], "payload": {"curation_id": str(db.curation["id"])}}
    context = SimpleNamespace(check=lambda: None, report=lambda _: None)
    return SimpleNamespace(db=db, remote=remote, plan=plan, selections=selections, job=job, context=context)


def test_publication_committed_before_remote_and_complete_is_idempotent(env):
    result = cj.execute(env.job, env.context)
    assert result["playlist_id"] == "created-id"
    assert result["track_count"] == 3
    assert env.db.curation["status"] == "ready"
    assert cj.execute(env.job, env.context) == result
    assert len(env.remote.calls) == len(env.selections) == len(env.db.revisions) == 1
    assert env.remote.calls[0][1] == ["b", "a", "b"]


def test_ambiguous_create_blocks_same_and_new_jobs(env):
    env.remote.error = TimeoutError("response lost")
    with pytest.raises(TimeoutError):
        cj.execute(env.job, env.context)
    env.remote.error = None
    for job_id in [env.job["id"], uuid4()]:
        with pytest.raises(RuntimeError, match="outcome is unknown"):
            cj.execute({**env.job, "id": job_id}, env.context)
    assert len(env.remote.calls) == len(env.selections) == 1
    assert env.db.curation["status"] == "failed"


def test_known_playlist_retries_exact_selection_even_in_new_job(env):
    env.db.curation["navidrome_playlist_id"] = "existing"
    env.db.commit()
    env.remote.error = TimeoutError("response lost")
    with pytest.raises(TimeoutError):
        cj.execute(env.job, env.context)
    env.plan["source_ids"] = ["different"]
    env.remote.error = None
    next_job = {**env.job, "id": uuid4()}
    result = cj.execute(next_job, env.context)
    assert cj.execute(next_job, env.context) == result
    assert result["playlist_id"] == "existing"
    assert env.remote.calls == [("Mix", ["b", "a", "b"], "existing")] * 2
    assert len(env.selections) == 1


def test_returned_id_survives_local_finalization_failure(env):
    env.db.fail_revision = True
    with pytest.raises(RuntimeError, match="local revision"):
        cj.execute(env.job, env.context)
    assert env.db.publication["playlist_id"] == "created-id"
    assert env.db.curation["navidrome_playlist_id"] == "created-id"
    env.db.fail_revision = False
    cj.execute(env.job, env.context)
    assert env.remote.calls[-1][2] == "created-id"
    assert len(env.db.revisions) == len(env.selections) == 1


def test_deleted_or_wrong_owner_does_not_publish(env):
    result = cj.execute({**env.job, "user_id": uuid4()}, env.context)
    assert result["deleted"]
    assert not env.remote.calls


def test_cancellation_after_preparing_keeps_resumable_plan(env):
    class Cancelled(BaseException):
        pass

    def report(update):
        if update["phase"] == "publishing":
            raise Cancelled("cancel requested")

    env.context.report = report
    with pytest.raises(Cancelled):
        cj.execute(env.job, env.context)
    assert env.db.curation["status"] == "failed"
    assert env.db.publication["state"] == "prepared"
    assert not env.remote.calls
    env.context.report = lambda _: None
    cj.execute({**env.job, "id": uuid4()}, env.context)
    assert len(env.selections) == 1


def test_mutation_guards_unresolved_publication(env):
    env.remote.error = TimeoutError()
    with pytest.raises(TimeoutError):
        cj.execute(env.job, env.context)
    with pytest.raises(ConflictError):
        cj.assert_mutable(env.db, env.db.curation["id"])
    with pytest.raises(ConflictError):
        cj.assert_mutable(env.db, env.db.curation["id"], deleting=True)
    cj.assert_mutable(env.db, env.db.curation["id"], deleting=True, delete_remote=False)


def test_lock_released_on_error(env):
    with pytest.raises(ValueError):
        with cj.locked(env.db.curation["id"]):
            raise ValueError("failure")
    sql = [e for e in env.db.events if "pg_advisory" in e]
    assert "pg_advisory_lock" in sql[0]
    assert "pg_advisory_unlock" in sql[-1]


def test_schedule_advances_only_after_enqueue(env, monkeypatch):
    import core.jobs.queue as jobs
    calls = []

    def enqueue(**kw):
        calls.append(kw)
        assert env.db.curation["due"]
        return {"id": uuid4(), "status": "queued"}

    monkeypatch.setattr(jobs, "enqueue", enqueue)
    assert cj.enqueue_due() == 1
    assert cj.enqueue_due() == 0
    assert calls[0]["dedupe_key"] == "curation:" + str(env.db.curation["id"])
    assert calls[0]["worker_type"] == "scheduled"
    assert calls[0]["payload"] == env.job["payload"]


def test_enqueue_failure_keeps_due_work(env, monkeypatch):
    import core.jobs.queue as jobs

    def fail(**kw):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(jobs, "enqueue", fail)
    with pytest.raises(RuntimeError):
        cj.enqueue_due()
    assert env.db.curation["due"]


def test_manual_enqueue_returns_contract_and_closes_draft_window(env, monkeypatch):
    import core.jobs.queue as jobs
    job_id = uuid4()
    monkeypatch.setattr(jobs, "enqueue", lambda **kw: {"id": job_id, "status": "queued", "existing": True})
    result = cj.enqueue(env.db.curation["id"], env.job["user_id"])
    assert result == {"job_id": str(job_id), "status": "queued", "curation_id": env.job["payload"]["curation_id"]}
    assert env.db.curation["status"] == "pending"


def test_routes_share_lock_and_refresh_is_202():
    tree = ast.parse((Path(__file__).resolve().parents[3] / "services/analysis/src/echora_analysis/main.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ["update_curation", "delete_curation"]:
        assert any(isinstance(n, ast.With) and any("curation_jobs.locked" in ast.unparse(i.context_expr)
                   for i in n.items) for n in ast.walk(functions[name]))
    decorator = ast.unparse(functions["refresh_curation"].decorator_list[0])
    assert "status_code=202" in decorator
    assert "enqueue" in ast.unparse(functions["_refresh_curation"])


def test_due_poll_skips_locked_curation_without_blocking(env, monkeypatch):
    import core.jobs.queue as jobs
    env.db.acquired = False
    calls = []
    monkeypatch.setattr(jobs, "enqueue", lambda **kw: calls.append(kw))
    assert cj.enqueue_due() == 0
    assert not calls
    assert env.db.curation["due"]


def test_schedule_crash_after_enqueue_reuses_dedupe_key(env, monkeypatch):
    import core.jobs.queue as jobs
    queued = {}
    calls = []

    def enqueue(**kw):
        key = (kw["user_id"], kw["dedupe_key"])
        calls.append(key)
        queued.setdefault(key, {"id": uuid4(), "status": "queued"})
        return queued[key]

    execute = env.db.execute
    fail_once = [True]

    def crash(sql, args=()):
        if "UPDATE curations" in sql and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("crash after enqueue committed")
        return execute(sql, args)

    monkeypatch.setattr(jobs, "enqueue", enqueue)
    monkeypatch.setattr(env.db, "execute", crash)
    with pytest.raises(RuntimeError, match="crash after enqueue"):
        cj.enqueue_due()
    assert env.db.curation["due"]
    assert cj.enqueue_due() == 1
    assert len(queued) == 1
    assert len(calls) == 2


def test_selection_failure_preserves_playlist_and_sets_failed(env, monkeypatch):
    def fail(*args):
        raise ValueError("No tracks meet this recipe")

    monkeypatch.setattr(cj, "_prepare", fail)
    with pytest.raises(ValueError, match="No tracks"):
        cj.execute(env.job, env.context)
    assert env.db.curation["status"] == "failed"
    assert env.db.publication is None
    assert not env.remote.calls


@pytest.mark.parametrize("deleted", [False, True])
def test_manual_enqueue_missing_or_wrong_owner_never_queues(env, monkeypatch, deleted):
    import core.jobs.queue as jobs

    calls = []
    monkeypatch.setattr(jobs, "enqueue", lambda **kw: calls.append(kw))
    curation_id = env.db.curation["id"]
    if deleted:
        env.db.curation = None
        env.db.commit()
    with pytest.raises(NotFoundError):
        cj.enqueue(curation_id, uuid4())
    assert not calls


def test_deleted_curation_job_is_noop(env):
    env.db.curation = None
    env.db.commit()
    assert cj.execute(env.job, env.context) == {
        "curation_id": env.job["payload"]["curation_id"], "deleted": True,
    }
    assert not env.remote.calls
    assert not env.selections


def test_known_playlist_can_be_deleted_after_uncertain_replacement(env):
    env.db.curation["navidrome_playlist_id"] = "existing"
    env.db.commit()
    env.remote.error = TimeoutError("response lost")
    with pytest.raises(TimeoutError):
        cj.execute(env.job, env.context)
    cj.assert_mutable(env.db, env.db.curation["id"], deleting=True)
    with pytest.raises(ConflictError):
        cj.assert_mutable(env.db, env.db.curation["id"])
