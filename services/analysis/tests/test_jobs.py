"""Unit tests plus isolated-schema PostgreSQL tests (opt in TEST_DATABASE_URL)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import os

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import pytest

from echora_analysis import jobs


def test_public_allowlist():
    row = {key: None for key in jobs.PUBLIC}
    row.update(id=uuid4(), created_at=datetime.now(timezone.utc), status='queued',
               payload={'password': 'secret'}, token=uuid4(), worker_id='private')
    public = jobs._public(row)
    assert public['id'] == public['job_id'] == str(row['id'])
    assert not {'payload', 'token', 'claim_token', 'worker_id'} & public.keys()
    assert public['existing'] is False


def test_context_fencing(monkeypatch):
    token = uuid4()
    context = jobs.JobContext({'id': str(uuid4()), 'claim_token': token})
    assert context.token == token
    for function in ('heartbeat', 'progress', 'finish'):
        monkeypatch.setattr(jobs, function, lambda *args, **kwargs: False)
    for call in (context.check, lambda: context.report({}), context.complete):
        with pytest.raises(jobs.JobCancelled):
            call()
    assert not issubclass(jobs.JobCancelled, Exception)


def test_input_validation():
    with pytest.raises(ValueError):
        jobs.finish(uuid4(), uuid4(), status='waiting')
    with pytest.raises(ValueError):
        jobs.expand(uuid4(), uuid4(), [None])
    with pytest.raises(ValueError):
        jobs.checkpoint(uuid4(), uuid4(), [])
    assert jobs._uuid(str(uuid4())) is not None


@pytest.fixture
def database(monkeypatch):
    url = os.getenv('TEST_DATABASE_URL')
    if not url:
        pytest.skip('TEST_DATABASE_URL is required for PostgreSQL queue tests')
    schema = 'test_jobs_' + uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    def connect():
        return psycopg.connect(url, options=f'-c search_path={schema}', row_factory=dict_row)
    monkeypatch.setattr(jobs, '_db', connect)
    try:
        with connect() as db:
            # Exercise the packaged migration, also available in deployed source trees.
            import runpy
            from unittest.mock import patch
            migration = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                           'alembic/versions/0034_jobs.py'))
            with patch('alembic.op.execute', side_effect=db.execute):
                migration['upgrade']()
                dismissal = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                              'alembic/versions/0049_job_dismissal.py'))
                dismissal['upgrade']()
        yield connect
    finally:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def ready(database, job_id):
    with database() as db:
        db.execute("UPDATE jobs SET available_at=now()-interval '1 second' WHERE id=%s", (job_id,))


def expire(database, job_id):
    with database() as db:
        db.execute("UPDATE jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (job_id,))


def test_owner_dedupe_and_public(database):
    owner, other, connection = str(uuid4()), str(uuid4()), str(uuid4())
    first = jobs.enqueue('analysis', 'analysis', owner, connection, {'secret': 'hidden'}, 'key')
    assert jobs.enqueue('analysis', 'analysis', owner, dedupe_key='key')['existing']
    assert jobs.enqueue('analysis', 'analysis', other, dedupe_key='key')['id'] != first['id']
    assert jobs.get_job(first['id'], other) is None
    assert jobs.cancel(first['id'], other) is None
    assert jobs.list_jobs(owner, connection) == [first]
    assert jobs.list_jobs(owner, str(uuid4())) == []
    assert jobs.cancel(first['id'], owner)['status'] == 'cancelled'
    assert jobs.list_jobs(owner, active_only=True) == []
    assert jobs.enqueue('analysis', 'analysis', owner, dedupe_key='key')['id'] != first['id']


def test_fencing_checkpoint_retry_and_error(database):
    owner = uuid4()
    public = jobs.enqueue('curation', 'curation', owner, payload={'seed': 1})
    job = jobs.claim('curation', 'worker')
    assert job['claim_token'] == job['token']
    assert jobs.claim('analysis', 'worker') is None
    assert not jobs.progress(public['id'], uuid4(), {'bad': True})
    assert jobs.checkpoint(public['id'], str(job['token']), {'selection': [1, 2]})
    assert jobs.progress(public['id'], job['token'], {'count': 2})
    assert jobs.fail(public['id'], job['token'], 'postgres://password-secret', True)
    assert not jobs.finish(public['id'], job['token'])
    assert jobs.claim('curation', 'worker') is None  # backoff
    ready(database, public['id'])
    second = jobs.claim('curation', 'worker2')
    assert second['payload'] == {'seed': 1, 'selection': [1, 2]}
    assert second['token'] != job['token']
    assert not jobs.checkpoint(public['id'], job['token'], {'selection': []})
    assert jobs.fail(public['id'], second['token'], 'secret', retryable=False)
    final = jobs.get_job(public['id'], owner)
    assert final['status'] == 'failed'
    assert final['error'] == 'Job execution failed.'
    assert 'secret' not in str(final)
    third = jobs.enqueue('curation', 'curation', owner, payload={})
    claim = jobs.claim('curation', 'worker')
    assert jobs.fail(third['id'], claim['token'], 'secret', retryable=False, note='Some songs failed in lyrics.')
    assert jobs.get_job(third['id'], owner)['error'] == 'Some songs failed in lyrics.'


def test_expired_lease_and_bounded_attempts(database):
    owner = uuid4()
    public = jobs.enqueue('analysis', 'analysis', owner)
    for attempt in range(1, 4):
        job = jobs.claim('analysis', 'worker')
        assert job['attempts'] == attempt
        expire(database, public['id'])
        assert not jobs.heartbeat(public['id'], job['token'])
        assert not jobs.finish(public['id'], job['token'])
        assert jobs.claim('analysis', 'replacement') is None
        ready(database, public['id'])
    assert jobs.get_job(public['id'], owner)['status'] == 'failed'
    assert jobs.claim('analysis', 'worker') is None


def test_running_cancellation_acknowledgement(database):
    owner = uuid4()
    public = jobs.enqueue('analysis', 'analysis', owner, dedupe_key='one')
    job = jobs.claim('analysis', 'worker')
    cancelled = jobs.cancel(public['id'], owner)
    assert cancelled['status'] == 'running' and cancelled['cancel_requested']
    with database() as db:
        assert db.execute('SELECT token FROM jobs WHERE id=%s', (job['id'],)).fetchone()['token'] == job['token']
    assert jobs.enqueue('analysis', 'analysis', owner, dedupe_key='one')['existing']
    assert not jobs.heartbeat(public['id'], job['token'])
    assert jobs.get_job(public['id'], owner)['status'] == 'cancelled'


def test_cancel_expiry(database):
    owner = uuid4()
    job = jobs.enqueue('analysis', 'analysis', owner)
    claim = jobs.claim('analysis', 'worker')
    jobs.cancel(job['id'], owner)
    expire(database, job['id'])
    assert jobs.claim('analysis', 'worker') is None
    assert jobs.get_job(job['id'], owner)['status'] == 'cancelled'
    assert not jobs.finish(job['id'], claim['token'])


def test_batches_aggregate_and_hide_children(database):
    owner, connection = uuid4(), uuid4()
    root = jobs.enqueue('analysis', 'analysis', owner, connection)
    claim = jobs.claim('analysis', 'worker')
    assert jobs.expand(root['id'], claim['token'], [{'batch': 1}, {'batch': 2}])
    assert not jobs.expand(root['id'], claim['token'], [])
    assert len(jobs.list_jobs(owner)) == 1
    assert jobs.get_job(root['id'], owner)['progress']['total'] == 2
    a, b = jobs.claim('analysis', 'a'), jobs.claim('analysis', 'b')
    assert a['parent_id'] == b['parent_id'] == claim['id']
    assert a['user_id'] == owner and a['connection_id'] == connection
    assert a['kind'] == 'analysis_batch'
    assert jobs.finish(a['id'], a['token'], {'ok': True})
    assert jobs.get_job(root['id'], owner)['status'] == 'waiting'
    assert jobs.fail(b['id'], b['token'], 'error', retryable=False)
    final = jobs.get_job(root['id'], owner)
    assert final['status'] == 'partial'
    assert final['progress']['completed'] == 1
    assert final['progress']['finished'] == final['progress']['total'] == 2
    assert final['message'].startswith('1 of 2 batches successful')


def test_cancel_batch_and_empty_batch(database):
    owner = uuid4()
    root = jobs.enqueue('analysis', 'analysis', owner)
    claim = jobs.claim('analysis', 'worker')
    assert jobs.expand(root['id'], claim['token'], [{}, {}])
    child = jobs.claim('analysis', 'worker')
    assert jobs.cancel(root['id'], owner)['status'] == 'waiting'
    assert not jobs.progress(child['id'], child['token'], {})
    assert jobs.get_job(root['id'], owner)['status'] == 'cancelled'
    root = jobs.enqueue('analysis', 'analysis', owner)
    claim = jobs.claim('analysis', 'worker')
    assert jobs.expand(root['id'], claim['token'], [])
    assert jobs.get_job(root['id'], owner)['status'] == 'complete'


def test_parallel_claim_and_dedupe(database):
    owner = uuid4()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: jobs.enqueue('analysis', 'analysis', owner,
                                                       dedupe_key='single'), range(6)))
    assert len({r['id'] for r in results}) == 1
    assert sum(not r['existing'] for r in results) == 1
    with ThreadPoolExecutor(max_workers=6) as pool:
        claims = list(pool.map(lambda i: jobs.claim('analysis', str(i)), range(6)))
    assert sum(c is not None for c in claims) == 1


def test_expand_rollback(database):
    owner = uuid4()
    root = jobs.enqueue('analysis', 'analysis', owner)
    claim = jobs.claim('analysis', 'worker')
    with pytest.raises(TypeError):
        jobs.expand(root['id'], claim['token'], [{'ok': 1}, {'bad': object()}])
    assert jobs.get_job(root['id'], owner)['status'] == 'running'
    with database() as db:
        assert db.execute('SELECT count(*) AS n FROM jobs WHERE parent_id=%s',
                          (root['id'],)).fetchone()['n'] == 0
    assert jobs.expand(root['id'], claim['token'], [{}, {}])
    children = [jobs.claim('analysis', 'worker') for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(lambda child: jobs.finish(child['id'], child['token']), children))
    assert jobs.get_job(root['id'], owner)['status'] == 'complete'


def test_skip_locked(database):
    owner = uuid4()
    first = jobs.enqueue('analysis', 'analysis', owner)
    second = jobs.enqueue('analysis', 'analysis', owner)
    with database() as db:
        db.execute('SELECT * FROM jobs WHERE id=%s FOR UPDATE', (first['id'],))
        claimed = jobs.claim('analysis', 'worker')
        assert claimed['job_id'] == second['id']


@pytest.mark.parametrize('outcome', ['empty', 'complete', 'failed', 'cancelled'])
def test_hum_parent_terminal_projection(database, outcome):
    owner = uuid4()
    parent = jobs.enqueue('hum_corpus', 'analysis', owner)
    with database() as db:
        db.execute("CREATE TABLE hum_corpora(id uuid PRIMARY KEY,user_id uuid,status text,error text,completed_at timestamptz)")
        db.execute("INSERT INTO hum_corpora(id,user_id,status) VALUES (%s,%s,'building')", (parent['id'], owner))
    claim = jobs.claim('analysis', 'worker')
    if outcome == 'empty':
        jobs.expand(claim['id'], claim['token'], [])
    elif outcome == 'cancelled':
        jobs.cancel(parent['id'], owner)
        assert not jobs.heartbeat(claim['id'], claim['token'])
    else:
        jobs.expand(claim['id'], claim['token'], [{'track_ids': ['a']}])
        child = jobs.claim('analysis', 'worker')
        jobs.finish(child['id'], child['token'], status=outcome)
    with database() as db:
        row = db.execute('SELECT * FROM hum_corpora WHERE id=%s', (parent['id'],)).fetchone()
    assert row['status'] == ('complete' if outcome in {'empty', 'complete'} else 'failed')
    assert row['completed_at'] is not None


def test_public_progress_snapshot(database):
    owner = uuid4()
    job = jobs.enqueue('import', 'analysis', owner)
    claim = jobs.claim('analysis', 'worker')
    jobs.progress(claim['id'], claim['token'], {'phase': 'muq', 'completed': 2, 'total': 3, 'token': 'hidden'})
    public = jobs.get_job(job['id'], owner)
    assert public['phase'] == 'muq'
    assert public['completed'] == 2 and public['total'] == 3
    assert 'token' not in public


def test_legacy_cancelled_progress_is_not_success():
    row = {key: None for key in jobs.PUBLIC}
    row.update(id=uuid4(), status='cancelled', progress={
        'unit': 'batches', 'complete': 1, 'cancelled': 29,
        'completed': 30, 'total': 30, 'message': 'Processed 30 of 30 batches'})
    public = jobs._public(row)
    assert public['completed'] == 1
    assert public['progress']['finished'] == 30
    assert public['message'] == '1 of 30 batches successful'


def test_cancelled_batches_finish_parent_without_counting_as_success(database):
    owner = uuid4()
    root = jobs.enqueue('navidrome_sync', 'analysis', owner)
    claim = jobs.claim('analysis', 'worker')
    jobs.expand(root['id'], claim['token'], [{'track_ids': ['a']}, {'track_ids': ['b']}])
    child = jobs.claim('analysis', 'worker')
    jobs.finish(child['id'], child['token'])
    final = jobs.cancel(root['id'], owner)
    assert final['status'] == 'cancelled'
    assert final['completed'] == 1 and final['total'] == 2
    assert final['progress']['finished'] == 2
    assert final['summary']['cancelled'] == 1


def test_batch_listing_is_owned_paginated_and_active_first(database):
    owner, stranger = uuid4(), uuid4()
    parent = jobs.enqueue('navidrome_sync', 'analysis', owner)
    claim = jobs.claim('analysis', 'worker')
    jobs.expand(parent['id'], claim['token'], [{'track_ids': ['a', 'b']}, {'track_ids': ['c']}])
    child = jobs.claim('analysis', 'worker')
    jobs.progress(child['id'], child['token'], {'phase': 'voice', 'completed': 1, 'total': 2, 'unit': 'tracks', 'message': 'Classifying vocals for Song'})
    assert jobs.list_batches(parent['id'], stranger) is None
    first = jobs.list_batches(parent['id'], owner, limit=1)
    assert first['total'] == 2 and len(first['batches']) == 1
    batch = first['batches'][0]
    assert batch['id'] == str(child['id']) and batch['phase'] == 'voice'
    assert batch['message'] == 'Classifying vocals for Song'
    assert batch['track_count'] == len(child['payload']['track_ids'])
    assert not {'payload', 'token', 'user_id', 'worker_id'} & batch.keys()
    next_page = jobs.list_batches(parent['id'], owner, limit=1, offset=1)
    assert next_page['batches'][0]['id'] != batch['id']
    numbers = {item['id']: item['batch_number'] for item in jobs.list_batches(parent['id'], owner)['batches']}
    jobs.finish(child['id'], child['token'])
    assert {item['id']: item['batch_number'] for item in jobs.list_batches(parent['id'], owner)['batches']} == numbers


def test_dismissal_is_owned_terminal_and_durable(database):
    owner, stranger = uuid4(), uuid4()
    job = jobs.enqueue('navidrome_sync', 'analysis', owner)
    assert jobs.dismiss(job['id'], stranger) is None
    with pytest.raises(ValueError, match='finished'):
        jobs.dismiss(job['id'], owner)
    claim = jobs.claim('analysis', 'worker')
    with pytest.raises(ValueError, match='finished'):
        jobs.dismiss(job['id'], owner)
    jobs.finish(claim['id'], claim['token'])
    dismissed = jobs.dismiss(job['id'], owner)
    assert dismissed['dismissed_at'] is not None
    assert dismissed['status'] == 'complete'
    assert jobs.dismiss(job['id'], owner)['dismissed_at'] == dismissed['dismissed_at']
    assert jobs.get_job(job['id'], owner)['dismissed_at'] == dismissed['dismissed_at']
    assert jobs.list_jobs(owner)[0]['dismissed_at'] == dismissed['dismissed_at']
    newer = jobs.enqueue('navidrome_sync', 'analysis', owner)
    assert newer['dismissed_at'] is None
    assert jobs.list_jobs(owner, active_only=True)[0]['id'] == newer['id']


def test_library_discovery_filters_before_limiting(database):
    owner = uuid4()
    sync = jobs.enqueue('navidrome_sync', 'analysis', owner)
    jobs.enqueue('recording_search', 'analysis', owner)
    jobs.enqueue('curation_refresh', 'scheduled', owner)
    assert [job['id'] for job in jobs.list_jobs(owner, library_only=True, limit=1)] == [sync['id']]


def test_historical_fusion_completion_uses_committed_summary():
    row = {key: None for key in jobs.PUBLIC}
    row.update(id=uuid4(), kind='semantic_fusion_build', status='complete',
               progress={'phase': 'complete', 'message': 'Storing 1029 fused vectors',
                         'completed': 0, 'total': 1029, 'unit': 'tracks'},
               summary={'fused': 1029, 'total': 1029})
    public = jobs._public(row)
    assert public['completed'] == public['total'] == 1029
    assert public['message'] == 'Stored 1029 fused vectors'
    assert public['unit'] == 'vectors'
    row['status'] = 'failed'
    assert jobs._public(row)['completed'] == 0


def test_library_discovery_includes_only_owned_connectionless_fusion(database):
    owner, stranger, connection, other_connection = uuid4(), uuid4(), uuid4(), uuid4()
    sync = jobs.enqueue('navidrome_sync', 'analysis', owner, connection)
    fusion = jobs.enqueue('semantic_fusion_build', 'analysis', owner)
    jobs.enqueue('navidrome_sync', 'analysis', owner)
    jobs.enqueue('semantic_fusion_build', 'analysis', owner, other_connection)
    jobs.enqueue('navidrome_sync', 'analysis', owner, other_connection)
    jobs.enqueue('semantic_fusion_build', 'analysis', stranger)
    jobs.enqueue('semantic_fusion_build', 'analysis', stranger, connection)

    expected = [fusion['id'], sync['id']]
    for active_only in (True, False):
        found = jobs.list_jobs(owner, connection, active_only=active_only, library_only=True)
        assert [job['id'] for job in found] == expected
    assert jobs.list_jobs(owner, connection) == [sync]
    assert jobs.list_jobs(owner, connection, library_only=True, limit=1)[0]['id'] == fusion['id']

    # History uses the same scope, including persisted acknowledgement.
    jobs.cancel(fusion['id'], owner)
    jobs.dismiss(fusion['id'], owner)
    assert [job['id'] for job in jobs.list_jobs(owner, connection, active_only=True, library_only=True)] == [sync['id']]
    history = jobs.list_jobs(owner, connection, library_only=True)
    assert [job['id'] for job in history] == expected
    assert history[0]['dismissed_at'] is not None


def test_context_honors_worker_lease_override(monkeypatch):
    from unittest.mock import Mock
    heartbeat = Mock(return_value=True)
    monkeypatch.setattr(jobs, 'heartbeat', heartbeat)
    context = jobs.JobContext({'id': 'job', 'claim_token': 'token'}, lease_seconds=45)
    context.check()
    heartbeat.assert_called_once_with('job', 'token', 45)
