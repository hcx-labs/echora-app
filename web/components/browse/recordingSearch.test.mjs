// Run: node --test web/components/browse/recordingSearch.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
async function load(path) {
  const source = stripTypeScriptTypes(readFileSync(new URL(path, import.meta.url), 'utf8'));
  return import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
}
const { recordingQualityLabel, recordingWarning, recordingTracks, confirmedMatch, curationHref, recordingMessage, MAX_RECORDING_BYTES, MAX_RECORDING_SECONDS } = await load('./recordingSearch.ts');
const { parseRecordingHandoff, resolveRecordingReference } = await load('../curate/recordingHandoff.ts');
const id = '12345678-1234-1234-1234-123456789abc';
const match = { track_id: id, title: 'Song', artist: 'Artist', album: 'Album', score: .8, offset_seconds: 10 };
test('only a complete identified top match enables curation', () => {
  for (const status of ['queued', 'running', 'complete', 'failed', 'cancelled', 'partial']) {
    for (const state of ['identified', 'ambiguous', 'no_match', 'insufficient_audio']) {
      const job = { status, result: { state, matches: [match, { ...match, track_id: 'second' }] } };
      assert.equal(confirmedMatch(job), status === 'complete' && state === 'identified' ? match : undefined);
      assert.ok(recordingMessage(job).length);
    }
  }
  assert.equal(confirmedMatch(null), undefined);
  assert.equal(confirmedMatch({ status: 'complete', result: { state: 'identified', matches: [] } }), undefined);
});
test('handoff encodes values and allows only UUID IDs and known intents', () => {
  assert.deepEqual(parseRecordingHandoff(curationHref(id, 'journey').split('?')[1]), { trackId: id, intent: 'journey' });
  assert.deepEqual(parseRecordingHandoff(curationHref(id, 'similar').split('?')[1]), { trackId: id, intent: 'similar' });
  for (const query of ['', 'recording_track=javascript:alert(1)&recording_intent=similar', `recording_track=${id}&recording_intent=redirect`, 'recording_track=../../admin&recording_intent=journey']) assert.equal(parseRecordingHandoff(query), null);
  const url = new URL(curationHref('id&recording_intent=evil', 'similar'), 'https://echora.test');
  assert.equal(url.searchParams.get('recording_track'), 'id&recording_intent=evil');
  assert.equal(url.searchParams.get('recording_intent'), 'similar');
});
test('reference resolution uses one authorized track lookup, not querystring metadata', async t => {
  const urls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    urls.push(url); assert.ok(options.signal);
    return { ok: true, json: async () => ({ tracks: [{ id, title: 'Authorized title', artist: 'Authorized artist', ignored: true }], total: 1 }) };
  });
  assert.deepEqual(await resolveRecordingReference(id, new AbortController().signal), { id, title: 'Authorized title', artist: 'Authorized artist', album: undefined });
  assert.equal(urls.length, 1);
  assert.equal(new URL(urls[0], 'https://echora.test').searchParams.get('track_id'), id);
});
test('unavailable and forbidden references fail closed', async t => {
  t.mock.method(globalThis, 'fetch', async () => ({ ok: true, json: async () => ({ tracks: [], total: 0 }) }));
  await assert.rejects(resolveRecordingReference(id, new AbortController().signal), /no longer available/);
  t.mock.method(globalThis, 'fetch', async () => ({ ok: false }));
  await assert.rejects(resolveRecordingReference(id, new AbortController().signal), /Could not load/);
});
test('recording limits and unprocessed audio constraints are explicit', () => {
  assert.equal(MAX_RECORDING_SECONDS, 20); assert.equal(MAX_RECORDING_BYTES, 8 * 1024 * 1024);
  const source = readFileSync(new URL('./RecordingSearchButton.tsx', import.meta.url), 'utf8');
  for (const constraint of ['echoCancellation', 'noiseSuppression', 'autoGainControl']) assert.ok(source.includes(`${constraint}: false`));
  assert.match(source, /body: audio/);
  assert.match(source, /response.status !== 202/);
  assert.match(source, /bytes > MAX_RECORDING_BYTES/);
  assert.match(source, /MAX_RECORDING_SECONDS \* 1000/);
});

test('identified recordings become labeled playable Browse results', () => {
  const playable = { ...match, source_id: 'source', cover_art: 'cover', duration_seconds: 240 };
  assert.deepEqual(recordingTracks({ status: 'complete', result: { state: 'identified', matches: [playable] } }),
    [{ id, title: 'Song', artist: 'Artist', album: 'Album', duration_seconds: 240, source_id: 'source', cover_art: 'cover', connection_id: undefined, matched_at_seconds: 10, matched_source: 'recording', recording_score: .8, recording_confirmed: true, recording_quality: 'high' }]);
  for (const state of ['no_match', 'insufficient_audio'])
    assert.deepEqual(recordingTracks({ status: 'complete', result: { state, matches: [playable] } }), []);
  assert.deepEqual(recordingTracks({ status: 'running', result: { state: 'identified', matches: [playable] } }), []);
});
test('recording uses the inline Browse capture flow, without a calibration fallback or modal', () => {
  const source = readFileSync(new URL('./RecordingSearchButton.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /RecordingCalibration|calibration_enabled|createPortal|role="dialog"/);
  assert.match(source, /availability\?\.recognition_enabled !== true/);
  assert.match(source, /callbacks.current.onResults\(tracks, recordingWarning\(value\)\)/);
  const browse = readFileSync(new URL('./BrowseLibrary.tsx', import.meta.url), 'utf8').replace(/\s+/g, ' ');
  assert.match(browse, /RecordingSearchButton ref=\{recordingRef\} onResults=\{showRecordingResults\} onError=\{setError\}/);
  assert.match(browse, /Recording results/);
});

test('ambiguous candidates retain rank and playback metadata without becoming confirmed', () => {
  const candidates = [match, { ...match, track_id: 'second', score: .77 }];
  const job = { status: 'complete', result: { state: 'ambiguous', matches: candidates } };
  const tracks = recordingTracks(job);
  assert.deepEqual(tracks.map(t => t.id), [id, 'second']);
  assert.deepEqual(tracks.map(t => t.recording_score), [.8, .77]);
  assert.ok(tracks.every(t => !t.recording_confirmed));
  assert.equal(confirmedMatch(job), undefined);
  assert.match(recordingWarning(job), /Possible matches/);
  assert.match(recordingWarning(job), /not confidence percentages/);
  const identified = { ...job, result: { ...job.result, state: 'identified' } };
  assert.deepEqual(recordingTracks(identified).map(t => t.recording_confirmed), [true, false]);
  assert.equal(recordingWarning(identified), '');
});
test('a single ambiguous position is not described as several recordings', () => {
  const job = { status: 'complete', result: { state: 'ambiguous', matches: [match] } };
  assert.equal(recordingTracks(job).length, 1);
  assert.match(recordingWarning(job), /song or its position/);
  assert.doesNotMatch(recordingWarning(job), /Possible matches|Several/);
  assert.equal(recordingWarning({ ...job, status: 'running' }), '');
  assert.equal(recordingWarning({ ...job, result: { state: 'ambiguous', matches: [] } }), '');
});

test('match quality labels preserve low-scoring candidates without claiming confidence', () => {
  for (const quality of ['high', 'possible', 'low']) {
    const job = { status: 'complete', result: { state: quality === 'high' ? 'identified' : 'ambiguous', matches: [{ ...match, match_quality: quality, score: .4 }] } };
    const tracks = recordingTracks(job);
    assert.equal(tracks.length, 1);
    assert.equal(tracks[0].recording_quality, quality);
    assert.equal(tracks[0].recording_score, .4);
  }
  assert.equal(recordingQualityLabel('high'), 'High-quality match');
  assert.equal(recordingQualityLabel('possible'), 'Possible match');
  assert.equal(recordingQualityLabel('low'), 'Low-quality match');
  assert.match(recordingWarning({ status: 'complete', result: { state: 'ambiguous', matches: [{ ...match, match_quality: 'low' }] } }), /Low-quality match evidence/);
});

const { resultConnection, canCurateRecording } = await load('./recordingSearch.ts');
const { RecordingAttempt } = await load('./recordingAttempt.ts');
test('discard and unmount cancel admitted jobs and reject late upload admissions', () => {
  const cancelled = [];
  const attempt = new RecordingAttempt(id => cancelled.push(id));
  const first = attempt.generation;
  assert.equal(attempt.admit(first, 'first'), true);
  attempt.invalidate();
  assert.deepEqual(cancelled, ['first']);
  assert.equal(attempt.admit(first, 'late'), false);
  assert.equal(attempt.jobId, '');
  assert.equal(attempt.admit(attempt.generation, 'new'), true);
  attempt.invalidate(); attempt.invalidate();
  assert.deepEqual(cancelled, ['first', 'late', 'new']);
});
test('recording connection is per result and never falls back; curation is top high confirmed only', () => {
  const tracks = recordingTracks({ status: 'complete', result: { state: 'identified', matches: [{ ...match, connection_id: 'other' }, match] } });
  assert.equal(resultConnection(tracks[0], 'selected'), 'other');
  assert.equal(resultConnection(tracks[1], 'selected'), '');
  assert.equal(resultConnection({}, 'selected'), 'selected');
  assert.equal(resultConnection({ matched_source: 'melody' }, 'selected'), 'selected');
  assert.equal(canCurateRecording(tracks[0], 0), true);
  assert.equal(canCurateRecording(tracks[0], 1), false);
  for (const recording_quality of ['possible', 'low']) assert.equal(canCurateRecording({ ...tracks[0], recording_quality }, 0), false);
  assert.equal(canCurateRecording({ ...tracks[0], recording_confirmed: false }, 0), false);
});
