import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const source = stripTypeScriptTypes(readFileSync(new URL('./recordingCalibration.ts', import.meta.url), 'utf8'));
const { calibrationRequest, isCalibrationOnly, validateCalibrationAudio, checkCalibrationAudio } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
test('calibration is explicit and never replaces available recognition', () => {
  assert.equal(isCalibrationOnly(null), false);
  assert.equal(isCalibrationOnly({ enabled: true }), false);
  assert.equal(isCalibrationOnly({ recognition_enabled: true, calibration_enabled: true }), false);
  assert.equal(isCalibrationOnly({ recognition_enabled: false, calibration_enabled: false }), false);
  assert.equal(isCalibrationOnly({ recognition_enabled: false, calibration_enabled: true }), true);
});
test('save sends raw audio with explicit consent and mutually exclusive labels', () => {
  const audio = new Blob(['clip'], { type: 'audio/webm' });
  const request = calibrationRequest(audio, 'track-id', false, '  speaker & noise?  ');
  const url = new URL(request.url, 'https://echora.test');
  assert.equal(url.searchParams.get('expected_track_id'), 'track-id');
  assert.equal(url.searchParams.has('notes'), false);
  assert.equal(decodeURIComponent(request.init.headers['X-Echora-Calibration-Notes']), 'speaker & noise?');
  assert.equal(url.searchParams.has('not_in_library'), false);
  assert.equal(request.init.method, 'POST');
  assert.equal(request.init.body, audio);
  assert.equal(request.init.headers['X-Echora-Calibration-Consent'], 'save-for-calibration-v1');
  assert.equal(request.init.headers['Content-Type'], 'audio/webm');
  const negative = new URL(calibrationRequest(audio, 'stale-track', true, '').url, 'https://echora.test');
  assert.equal(negative.searchParams.get('not_in_library'), 'true');
  assert.equal(negative.searchParams.has('expected_track_id'), false);
  assert.equal(negative.searchParams.has('notes'), false);
  assert.throws(() => calibrationRequest(audio, undefined, false, ''), /Choose the actual/);
  assert.throws(() => calibrationRequest(audio, 'track-id', false, 'x'.repeat(301)), /300 characters/);
  const unicode = calibrationRequest(audio, 'track-id', false, '音楽 · speaker');
  assert.equal(decodeURIComponent(unicode.init.headers['X-Echora-Calibration-Notes']), '音楽 · speaker');
});
test('clip boundaries reject empty, oversized, overlong and unmeasurable audio', () => {
  assert.doesNotThrow(() => validateCalibrationAudio(8 * 1024 * 1024, 20));
  for (const [size, seconds] of [[0, 1], [8 * 1024 * 1024 + 1, 1], [10, 20.01], [10, 0], [10, Infinity], [10, NaN]]) assert.throws(() => validateCalibrationAudio(size, seconds));
});
test('duration is decoded locally and audio context is released even on rejection', async t => {
  let duration = 2, closed = 0;
  t.mock.method(globalThis, 'fetch', () => { throw new Error('Unexpected upload'); });
  const previous = globalThis.AudioContext;
  globalThis.AudioContext = class { async decodeAudioData() { return { duration }; } async close() { closed++; } };
  try {
    await checkCalibrationAudio(new Blob(['clip']));
    duration = 21;
    await assert.rejects(checkCalibrationAudio(new Blob(['clip'])), /20 seconds/);
    assert.equal(closed, 2);
  } finally { if (previous) globalThis.AudioContext = previous; else delete globalThis.AudioContext; }
});
test('collection UI exposes local preview, disclosure, explicit save and metadata deletion only', () => {
  const panel = readFileSync(new URL('./RecordingCalibration.tsx', import.meta.url), 'utf8');
  for (const text of ['Save calibration clip', '7 days', 'captured speech', 'server disk', 'TrackReferencePicker', 'Not in my library', 'URL.revokeObjectURL', 'method: "DELETE"', 'response.status !== 201', 'response.status !== 204']) assert.ok(panel.includes(text), text);
  assert.match(panel, /onClick=\{\(\) => void save\(\)\}/);
  assert.doesNotMatch(panel, /curationHref|TransitionLink|\/search|download=/);
  const recorder = readFileSync(new URL('./RecordingSearchButton.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(recorder, /RecordingCalibration|checkCalibrationAudio|isCalibrationOnly/);
});
