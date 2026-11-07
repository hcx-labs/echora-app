"""Separate analysis and scheduled workers: python -m echora_analysis.worker analysis."""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import signal
import socket
import threading
import tempfile
import sys
import time
import uuid

from . import jobs
from .settings import get_settings

logger = logging.getLogger(__name__)


def _has_failures(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            (key == "failed" or key.endswith("_failed")) and isinstance(item, (int, float)) and item > 0
            or _has_failures(item)
            for key, item in value.items()
        )
    return False


# Translations can be filled in later on their own; one failed request shouldn't re-run a batch.
RETRY_EXEMPT = {'translations'}


def _failed_stages(summary: dict) -> list[str]:
    return [key.replace('_', ' ') for key, item in summary.items()
            if (key == 'failed' and isinstance(item, (int, float)) and item > 0) or (key != 'failed' and _has_failures(item))]


def _failure_note(stages: list[str], retrying: bool) -> str:
    # Built from summary keys only, never exception text.
    named = [s for s in stages if s != 'failed'] or ['songs']
    return f"Some songs failed in {', '.join(named)}." + (' Retrying.' if retrying else '')


def execute(job: dict, work_directory: str | None = None, supervisor_pid: int | None = None,
            lease_seconds: int | None = None) -> None:
    """Run one claim in an isolated process so model memory dies with the claim."""
    # Pipeline subprocesses inherit this group, allowing the supervisor to stop all work.
    if hasattr(os, "setsid"):
        os.setsid()
    def cancelled(*_):
        raise jobs.JobCancelled()
    signal.signal(signal.SIGTERM, cancelled)
    if sys.platform == 'linux' and supervisor_pid is not None:
        # Stop an orphaned executor when its supervisor is killed, even by SIGKILL.
        import ctypes
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise RuntimeError('Could not install parent-death signal')
        if os.getppid() != supervisor_pid:
            return
    if work_directory is not None:
        os.environ['TMPDIR'] = work_directory
        tempfile.tempdir = work_directory
    previous_job_id = os.environ.get('ECHORA_JOB_ID')
    os.environ['ECHORA_JOB_ID'] = str(job['id'])
    context = jobs.JobContext(job, lease_seconds=lease_seconds)
    try:
        context.check()
        if job["worker_type"] == "analysis":
            from .analysis_jobs import execute as handler
        else:
            from .curation_jobs import execute as handler
        summary = handler(job, context)
        # None means the scan expanded into durable child batches.
        if summary is not None:
            partial = _has_failures(summary)
            retry = [s for s in _failed_stages(summary) if s not in RETRY_EXEMPT]
            if retry and job.get('attempts', 1) < job.get('max_attempts', get_settings().job_max_attempts):
                # Per-song failures are swallowed by pipelines to preserve the rest
                # of the batch. Retry that batch using its committed artifact plan.
                note = _failure_note(retry, retrying=True)
                context.report({'message': note, 'summary': summary})
                jobs.fail(job['id'], job['claim_token'], 'Incomplete analysis', retryable=True, note=note)
            else:
                context.complete(summary, status="partial" if partial else "complete")
    except jobs.JobCancelled:
        logger.info("Job %s cancelled or claim lost", job["id"])
    except Exception:
        # Do not log provider exception strings: they can contain signed URLs/credentials.
        logger.error("Job %s execution failed", job["id"])
        jobs.fail(job["id"], job["claim_token"], "Execution failed", retryable=True)
    finally:
        if previous_job_id is None:
            os.environ.pop('ECHORA_JOB_ID', None)
        else:
            os.environ['ECHORA_JOB_ID'] = previous_job_id


def _terminate(process) -> None:
    if not process.is_alive():
        process.join()
        return
    try:
        if hasattr(os, "killpg") and os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    process.join(timeout=get_settings().worker_shutdown_grace_seconds)
    if process.is_alive():
        try:
            if hasattr(os, "killpg") and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.join(timeout=get_settings().worker_shutdown_grace_seconds)


def supervise(job: dict, stop: threading.Event, lease_seconds: int) -> None:
    # The supervisor owns cleanup even when an executor cannot run its finally blocks.
    directory = tempfile.TemporaryDirectory(prefix='echora-worker-')
    process = multiprocessing.get_context("spawn").Process(
        target=execute, args=(job, directory.name, os.getpid(), lease_seconds))
    try:
        process.start()
    except BaseException:
        directory.cleanup()
        raise
    heartbeat_interval = min(get_settings().worker_heartbeat_seconds, lease_seconds / 3)
    try:
        while process.is_alive():
            if stop.wait(heartbeat_interval):
                _terminate(process)
                jobs.fail(job["id"], job["claim_token"], "Worker shutdown", retryable=True)
                return
            try:
                owned = jobs.heartbeat(job["id"], job["claim_token"], lease_seconds)
            except Exception:
                # Fail closed: don't keep writing artifacts without a live claim.
                logger.error("Job %s heartbeat failed", job["id"])
                _terminate(process)
                return
            if not owned:
                _terminate(process)
                return
        process.join()
        if process.exitcode:
            jobs.fail(job["id"], job["claim_token"], "Worker process exited", retryable=True)
    finally:
        _terminate(process)
        directory.cleanup()


def run(worker_type: str, *, once: bool = False, poll_seconds: float | None = None, lease_seconds: int | None = None) -> None:
    settings = get_settings()
    poll_seconds = settings.worker_poll_seconds if poll_seconds is None else poll_seconds
    lease_seconds = settings.worker_lease_seconds if lease_seconds is None else lease_seconds
    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"
    next_schedule_check = 0.0
    next_recording_cleanup = 0.0
    while not stop.is_set():
        try:
            if worker_type == "analysis" and time.monotonic() >= next_recording_cleanup:
                from .recording_search import cleanup
                from .recording_calibration import cleanup as cleanup_calibration
                next_recording_cleanup = time.monotonic() + settings.worker_cleanup_interval_seconds
                for maintenance in (cleanup, cleanup_calibration):
                    try:
                        maintenance()
                    except Exception:
                        # A bad retained file must not block unrelated analysis.
                        # Avoid exception text, which can expose private paths.
                        logger.error("Recording retention maintenance failed; will retry")
            if worker_type == "scheduled" and time.monotonic() >= next_schedule_check:
                from .curation_jobs import enqueue_due
                enqueue_due()
                next_schedule_check = time.monotonic() + settings.worker_schedule_check_seconds
            job = jobs.claim(worker_type, worker_id, lease_seconds)
            if job:
                supervise(job, stop, lease_seconds)
            elif not once:
                stop.wait(poll_seconds)
        except Exception:
            logger.error("Worker database or scheduling operation failed")
            if once:
                raise
            stop.wait(poll_seconds)
        if once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_type", choices=("analysis", "scheduled"))
    parser.add_argument("--once", action="store_true", help="Claim at most one job, then exit")
    parser.add_argument("--poll-seconds", type=float, default=get_settings().worker_poll_seconds)
    parser.add_argument("--lease-seconds", type=int, default=get_settings().worker_lease_seconds)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.lease_seconds < 15:
        parser.error("poll seconds must be positive and lease seconds must be at least 15")
    logging.basicConfig(level=logging.INFO)
    run(args.worker_type, once=args.once, poll_seconds=args.poll_seconds, lease_seconds=args.lease_seconds)


if __name__ == "__main__":
    main()
