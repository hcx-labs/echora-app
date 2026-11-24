"""Detect compressed timestamp sequences without banning repeated lyrics."""
import re


def compressed_timing(segments, start, end):
    """Conservative heuristic, not a confidence score or a language detector.

    Require several distinct lines, many whitespace-delimited words, and a
    whole transcript compressed into a small fraction of its audio window.
    This intentionally does not classify unsegmented scripts or short ad-libs.
    """
    if len(segments) < 4 or end <= start:
        return None
    texts = [str(s['text']).strip() for s in segments]
    words = sum(len(text.split()) for text in texts)
    distinct = len({re.sub(r'\W+', ' ', t.casefold()).strip() for t in texts})
    span = (max(s['end_ms'] for s in segments)-min(s['start_ms'] for s in segments))/1000
    if words < 24 or distinct/len(texts) < .5 or span <= 0:
        return None
    if span/(end-start) > .35 or words/span < 6:
        return None
    return {'reason': 'compressed_timestamps', 'words': words, 'span_seconds': span,
            'window_seconds': end-start, 'words_per_second': words/span,
            'distinct_lines': distinct}


def apply_aligned_times(source, aligned, start, end):
    """Transfer only timestamps. Require one ordered alignment per original line."""
    if len(source) != len(aligned):
        raise ValueError('Timing alignment changed the number of lines')
    result = []
    previous = -1
    for original, line in zip(source, aligned):
        a, b = line['start_ms'], line['end_ms']
        if not 0 <= a < b <= round((end-start)*1000)+10 or a >= round((end-start)*1000) or a < previous:
            raise ValueError('Invalid timing alignment interval')
        if original['text'].strip() != str(line['text']).strip():
            raise ValueError('Timing alignment changed line text or order')
        previous = a
        result.append({**original, 'start_ms': round(start*1000)+a,
                       'end_ms': min(round(end*1000), round(start*1000)+b)})
    return result
