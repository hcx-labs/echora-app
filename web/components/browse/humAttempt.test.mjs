import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./humAttempt.ts', import.meta.url), 'utf8'));
const { HumAttempt } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
function media() {
  let stopped = 0;
  return { getTracks: () => [{ stop: () => stopped++ }], get stopped() { return stopped; } };
}

test('late microphone permission after cancellation or unmount immediately releases tracks', async () => {
  const attempt = new HumAttempt();
  const permission = deferred();
  const input = media();
  const admitted = permission.promise.then(stream => attempt.attachStream(stream));
  attempt.cancel();
  permission.resolve(input);
  assert.equal(await admitted, false);
  assert.equal(input.stopped, 1);
});

test('cancel detaches recorder handlers before stopping and releases its stream once', () => {
  const attempt = new HumAttempt();
  const input = media();
  attempt.attachStream(input);
  let stops = 0;
  attempt.recorder = {
    state: 'recording',
    onstop: () => assert.fail('cancel must not submit a recording'),
    ondataavailable: () => {}, onerror: () => {},
    stop() { stops++; assert.equal(this.onstop, null); assert.equal(this.ondataavailable, null); assert.equal(this.onerror, null); },
  };
  attempt.cancel(); attempt.cancel();
  assert.equal(stops, 1);
  assert.equal(input.stopped, 1);
  assert.equal(attempt.signal.aborted, true);
});

test('late successful and failed responses cannot publish after a newer operation starts', async () => {
  for (const fails of [false, true]) {
    const old = new HumAttempt();
    const response = deferred();
    const callbacks = [];
    const pending = (async () => {
      try { await response.promise; if (old.current) callbacks.push('results'); }
      catch { if (old.current) callbacks.push('error'); }
      finally { if (old.current) callbacks.push('idle'); }
    })();
    old.cancel();
    const newer = new HumAttempt();
    response.resolve(fails ? Promise.reject(new Error('late error')) : []);
    await pending;
    assert.deepEqual(callbacks, []);
    assert.equal(newer.current, true);
  }
});

test('normal stop releases the microphone without aborting the subsequent upload', () => {
  const attempt = new HumAttempt();
  const input = media();
  assert.equal(attempt.attachStream(input), true);
  attempt.release();
  assert.equal(input.stopped, 1);
  assert.equal(attempt.current, true);
});

test('Browse wires both starts and all reset paths to cancellation', () => {
  const browse = readFileSync(new URL('./BrowseLibrary.tsx', import.meta.url), 'utf8').replace(/\s+/g, ' ');
  assert.match(browse, /function invalidateSearches\(\) \{ recordingRef.current\?\.cancel\(\); humRef.current\?\.cancel\(\); \}/);
  assert.match(browse, /function resetResults\(\) \{ invalidateSearches\(\)/);
  assert.match(browse, /HumSearchButton ref=\{humRef\} onStart=\{invalidateSearches\}/);
  assert.match(browse, /RecordingSearchButton[^>]+onStart=\{invalidateSearches\}/);
  assert.match(browse, /invalidateSearches\(\); setArtistQuery/);
  assert.match(browse, /invalidateSearches\(\); setAlbumQuery/);
  const hum = readFileSync(new URL('./HumSearchButton.tsx', import.meta.url), 'utf8');
  assert.match(hum, /await response.json\(\) : null;\s+if \(!attempt.current\) return;/);
  assert.match(hum, /signal: attempt.signal/g);
  assert.match(hum, /useEffect\(\(\) => \(\) => attemptRef.current\?\.cancel\(\), \[\]\)/);
});
