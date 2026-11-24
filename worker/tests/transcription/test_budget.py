import pytest
from worker.transcription.budget import decode_song
from worker.transcription.recovery import WindowDecodeError
from worker.transcription.config import transcription_enabled
from unittest.mock import MagicMock


def test_every_window_attempted_before_retries_and_limits_shared():
    calls=[]
    ranges=[(i*48,i*48+60) for i in range(10)]
    def decode(a,b,limit):
        calls.append((a,b,limit))
        raise WindowDecodeError('loop',tokens=limit)
    results,gaps,budget=decode_song(ranges,decode,lambda a,b:None)
    assert calls[:10]==[(a,b,2048) for a,b in ranges]
    assert len(calls)==14
    assert budget['calls']==4 and budget['tokens']==4096
    assert not any(results) and len(gaps)==20


def test_failed_and_successful_retry_tokens_both_count_and_merge_midpoints():
    def decode(a,b,limit):
        if limit==2048: raise WindowDecodeError('bad',tokens=2000)
        return [{'start_ms':a*1000,'end_ms':b*1000,'text':'line'}],123
    results,gaps,budget=decode_song([(0,60)],decode,lambda a,b:None)
    assert len(results[0])==2 and not gaps
    assert budget['tokens']==246 and budget['calls']==2


def test_evidence_declines_only_retries():
    calls=[]
    def decode(a,b,limit):
        calls.append(limit);raise WindowDecodeError('bad',tokens=80)
    _,gaps,budget=decode_song([(0,60)],decode,lambda a,b:{'reason':'low_vocal_evidence'})
    assert calls==[2048] and budget['calls']==0
    assert gaps[0]['retry_evidence']['reason']=='low_vocal_evidence'


def test_runtime_errors_propagate():
    def decode(a,b,limit): raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError): decode_song([(0,60)],decode,lambda a,b:None)


@pytest.mark.parametrize('row,expected',[(None,False),((False,),False),((True,),True)])
def test_transcription_opt_in(row,expected):
    conn=MagicMock();conn.cursor.return_value.__enter__.return_value.fetchone.return_value=row
    assert transcription_enabled(conn) is expected
