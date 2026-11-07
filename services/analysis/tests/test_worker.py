from unittest.mock import Mock
from uuid import uuid4

import pytest

from echora_analysis import analysis_jobs, jobs, worker


@pytest.fixture
def execution(monkeypatch):
    monkeypatch.setattr(worker.os, 'setsid', Mock())
    monkeypatch.setattr(worker.signal, 'signal', Mock())
    context = Mock()
    monkeypatch.setattr(jobs, 'JobContext', Mock(return_value=context))
    failure = Mock()
    monkeypatch.setattr(jobs, 'fail', failure)
    job = {'id': uuid4(), 'claim_token': uuid4(), 'worker_type': 'analysis',
           'attempts': 1, 'max_attempts': 3}
    return job, context, failure


def test_expansion_is_not_finished_twice(execution, monkeypatch):
    job, context, failure = execution
    monkeypatch.setattr(analysis_jobs, 'execute', Mock(return_value=None))
    worker.execute(job)
    context.complete.assert_not_called()
    failure.assert_not_called()


def test_partial_batch_retries_then_reports_partial(execution, monkeypatch):
    job, context, failure = execution
    summary = {'lyrics': {'failed': 1}}
    monkeypatch.setattr(analysis_jobs, 'execute', Mock(return_value=summary))
    worker.execute(job)
    failure.assert_called_once()
    context.complete.assert_not_called()
    job['attempts'] = 3
    worker.execute(job)
    context.complete.assert_called_once_with(summary, status='partial')


def test_failed_translation_finishes_partial_without_retry(execution, monkeypatch):
    job, context, failure = execution
    summary = {'karaoke': {'failed': 0, 'aligned': 32}, 'translations': {'failed': 1, 'translated': 9}}
    monkeypatch.setattr(analysis_jobs, 'execute', Mock(return_value=summary))
    worker.execute(job)
    failure.assert_not_called()
    context.complete.assert_called_once_with(summary, status='partial')


def test_retry_names_the_failed_stages(execution, monkeypatch):
    job, context, failure = execution
    summary = {'failed': 0, 'karaoke': {'failed': 2}, 'translations': {'failed': 1}}
    monkeypatch.setattr(analysis_jobs, 'execute', Mock(return_value=summary))
    worker.execute(job)
    assert failure.call_args.kwargs['note'] == 'Some songs failed in karaoke. Retrying.'


def test_cancel_does_not_retry(execution, monkeypatch):
    job, context, failure = execution
    monkeypatch.setattr(analysis_jobs, 'execute', Mock(side_effect=jobs.JobCancelled()))
    worker.execute(job)
    context.complete.assert_not_called()
    failure.assert_not_called()


def test_failure_never_persists_provider_exception(execution, monkeypatch):
    job, context, failure = execution
    monkeypatch.setattr(analysis_jobs, 'execute', Mock(side_effect=RuntimeError('password=secret')))
    worker.execute(job)
    assert 'secret' not in repr(failure.call_args)
    context.complete.assert_not_called()


def test_success_is_terminal(execution, monkeypatch):
    job, context, failure = execution
    summary = {'failed': 0, 'embedded': 3}
    monkeypatch.setattr(analysis_jobs, 'execute', Mock(return_value=summary))
    worker.execute(job)
    context.complete.assert_called_once_with(summary, status='complete')
    failure.assert_not_called()


@pytest.mark.parametrize("failing", ["search", "calibration"])
def test_retention_failure_does_not_block_job_claims(monkeypatch, caplog, failing):
    from echora_analysis import recording_search, recording_calibration
    search_cleanup = Mock(side_effect=OSError("private-path") if failing == "search" else None)
    calibration_cleanup = Mock(side_effect=OSError("private-path") if failing == "calibration" else None)
    monkeypatch.setattr(recording_search, "cleanup", search_cleanup)
    monkeypatch.setattr(recording_calibration, "cleanup", calibration_cleanup)
    monkeypatch.setattr(worker.signal, "signal", Mock())
    claim = Mock(return_value=None)
    monkeypatch.setattr(jobs, "claim", claim)
    worker.run("analysis", once=True)
    claim.assert_called_once()
    search_cleanup.assert_called_once()
    calibration_cleanup.assert_called_once()
    assert "retention maintenance failed" in caplog.text
    assert "private-path" not in caplog.text
