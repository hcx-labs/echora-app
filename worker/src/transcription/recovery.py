"""Bounded retries with explicit reporting of unresolved audio regions."""
import re
import json
import logging
import tempfile


def diagnostic_writer(track_id):
    """Keep failed-window text in private, container-local files for investigation."""
    def write(attempt):
        if 'error' not in attempt:
            return
        try:
            with tempfile.NamedTemporaryFile(mode='w', prefix=f'echora-transcription-{track_id}-', suffix='.json', delete=False) as handle:
                json.dump(attempt, handle, ensure_ascii=False)
            logging.getLogger(__name__).warning('Transcription window failed for %s at %.2f-%.2fs; diagnostics: %s', track_id, attempt['start'], attempt['end'], handle.name)
        except OSError:
            logging.getLogger(__name__).exception('Could not save transcription diagnostics for %s', track_id)
    return write


def stalled_segments(text):
    matches = re.findall(r'\[(\d+(?:\.\d+)?)\]\[(?:S\d+|MULTI)\]([^\[\]]*)\[(\d+(?:\.\d+)?)\]', text)
    if len(matches) < 4:
        return False
    tail = [(float(a), float(b), t.strip()) for a,t,b in matches[-4:]]
    return len(set(tail)) == 1


class WindowDecodeError(ValueError):
    def __init__(self, message, tokens=0):
        super().__init__(message)
        self.tokens = tokens


def recover_window(decode, start, end, check=lambda: None, depth=0, on_unresolved=None, retry_policy=None):
    """Retry failed windows as overlapping halves, down to 7.5 seconds.

    Never treat a decoding failure as silence. An optional callback records
    unresolved leaves, allowing usable regions to survive without claiming
    complete transcription. Without that callback, failures still propagate.
    """
    check()
    try:
        return decode(start, end)
    except WindowDecodeError as error:
        evidence = retry_policy(start, end) if retry_policy is not None else None
        if evidence and on_unresolved is not None:
            on_unresolved({'start_ms':round(start*1000), 'end_ms':round(end*1000),
                           'reason':str(error), 'status':'unresolved', 'retry_evidence':evidence})
            return []
        if end-start <= 7.5 or depth >= 4:
            if on_unresolved is None:
                raise
            on_unresolved({'start_ms':round(start*1000), 'end_ms':round(end*1000),
                           'reason':str(error), 'status':'unresolved'})
            return []
    middle = (start+end)/2
    context = min(2.0, (end-start)/10)
    left = recover_window(decode, start, middle+context, check, depth+1, on_unresolved, retry_policy)
    right = recover_window(decode, middle-context, end, check, depth+1, on_unresolved, retry_policy)
    return ([s for s in left if (s['start_ms']+s['end_ms'])/2000 < middle]
            + [s for s in right if (s['start_ms']+s['end_ms'])/2000 >= middle])
