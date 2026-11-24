import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./sessionActivity.ts', import.meta.url), 'utf8'));
const { SessionActivity, BOOTSTRAP_SAFETY_TIMEOUT_MS } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const info = overrides => ({ renewed: false, expires_at: '2030-01-08T00:00:00Z', absolute_expires_at: '2030-01-31T00:00:00Z', renew_after_seconds: 120, activity_check_seconds: 10, activity_window_seconds: 30, request_timeout_seconds: 2, ...overrides });
const response = (body = info(), status = 200) => ({ status, ok: status === 200, json: async () => body });
const flush = async () => { for (let i = 0; i < 16; i++) await Promise.resolve(); };
function harness(options = {}) {
  let now = 0, visible = true, expired = 0, serial = 0;
  const timers = new Map(), calls = [];
  const shared = options.shared || { value: null };
  let handler = options.handler || (() => Promise.resolve(response(options.info || info())));
  const tracker = new SessionActivity({
    now: () => now, visible: () => visible,
    request: (activity, signal) => { calls.push({ activity, signal, at: now }); return handler(activity, signal); },
    schedule: (callback, delay) => { const id = ++serial; timers.set(id, { callback, at: now + delay }); return id; },
    cancel: id => timers.delete(id), expired: () => expired++,
    readCadence: () => shared.value, writeCadence: value => { shared.value = value; },
    exclusive: async work => work(),
  });
  return { tracker, calls, timers, shared, get expired() { return expired; },
    setVisible(value) { visible = value; }, setHandler(value) { handler = value; },
    async advance(ms) {
      const end = now + ms;
      while (true) {
        const next = [...timers.entries()].filter(([, timer]) => timer.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        now = next[1].at; timers.delete(next[0]); next[1].callback(); await flush();
      }
      now = end; await flush();
    },
    async start() { tracker.start(); await flush(); },
  };
}
const sample = (currentTime, overrides = {}) => ({ source: 'song', currentTime, paused: false, seeking: false, readyState: 4, ...overrides });

test('bootstrap once; idle checks and passive polling do not renew', async () => {
  const h = harness(); await h.start(); h.tracker.start();
  await h.advance(3_600_000);
  assert.equal(h.calls.length, 1); assert.equal(h.calls[0].activity, false);
  h.tracker.gesture(false); await flush();
  assert.equal(h.calls.length, 1);
  h.tracker.stop(); assert.equal(h.timers.size, 0);
});

test('trusted visible interaction renews only when server cadence is due', async () => {
  const h = harness(); await h.start();
  h.tracker.gesture(true); await h.advance(110_000);
  assert.equal(h.calls.length, 1);
  h.tracker.gesture(true); await h.advance(10_000);
  assert.equal(h.calls.length, 2); assert.equal(h.calls[1].at, 120_000);
  assert.equal(h.calls[1].activity, true);
  for (let i = 0; i < 11; i++) { h.tracker.gesture(true); await h.advance(10_000); }
  assert.equal(h.calls.length, 2);
  h.tracker.gesture(true); await h.advance(10_000);
  assert.equal(h.calls.length, 3);
});

test('hidden forgotten tab ignores gestures and old foreground activity; returning counts', async () => {
  const h = harness({ info: info({ renew_after_seconds: 10 }) }); await h.start();
  h.tracker.gesture(true); h.setVisible(false);
  await h.advance(10_000); h.tracker.gesture(true); await h.advance(100_000);
  assert.equal(h.calls.length, 1);
  h.setVisible(true); h.tracker.gesture(true); await flush();
  assert.equal(h.calls.length, 2);
});

test('genuinely advancing background playback counts; stalled and paused playback age out', async () => {
  const h = harness({ info: info({ renew_after_seconds: 10 }) }); await h.start(); h.setVisible(false);
  h.tracker.playback(sample(0)); await h.advance(10_000);
  assert.equal(h.calls.length, 1);
  h.tracker.playback(sample(1)); await flush(); assert.equal(h.calls.length, 2);
  for (let i = 2; i < 7; i++) { await h.advance(10_000); h.tracker.playback(sample(i)); await flush(); }
  assert.ok(h.calls.length >= 6);
  h.tracker.playback(sample(6)); await h.advance(40_000);
  const stalledCount = h.calls.length;
  for (let i = 0; i < 10; i++) { h.tracker.playback(sample(6)); await h.advance(10_000); }
  assert.equal(h.calls.length, stalledCount);
  for (let i = 7; i < 12; i++) { h.tracker.playback(sample(i, { paused: true })); await h.advance(10_000); }
  assert.equal(h.calls.length, stalledCount);
});

test('seeking, buffering and a new source do not constitute playback progress', async () => {
  const h = harness({ info: info({ renew_after_seconds: 0 }) }); await h.start(); h.setVisible(false); await h.advance(10_000);
  h.tracker.playback(sample(1)); h.tracker.playback(sample(90, { seeking: true })); h.tracker.playback(sample(90));
  h.tracker.playback(sample(91, { readyState: 2 })); h.tracker.playback(sample(92));
  h.tracker.playback(sample(100, { source: 'different' })); await flush();
  assert.equal(h.calls.length, 1);
});

test('shared server deadlines survive tabs/reloads, even a stale bootstrap response', async () => {
  const shared = { value: null };
  const a = harness({ shared, info: info({ renew_after_seconds: 10 }) }); await a.start();
  await a.advance(10_000); a.setHandler(() => Promise.resolve(response(info({ renewed: true, renew_after_seconds: 3600 }))));
  a.tracker.gesture(true); await flush();
  const b = harness({ shared, info: info({ renew_after_seconds: 0 }) }); await b.start();
  await b.advance(10_000); b.tracker.gesture(true); await flush();
  assert.equal(b.calls.length, 1); assert.equal(shared.value.dueAt, 3_610_000);
});

test('request errors are throttled; at most one request in flight; stop aborts and ignores late 401', async () => {
  const h = harness({ info: info({ renew_after_seconds: 0 }) }); await h.start(); await h.advance(10_000);
  h.setHandler(() => Promise.reject(new Error('offline')));
  h.tracker.gesture(true); await flush(); assert.equal(h.calls.length, 2);
  h.tracker.gesture(true); h.tracker.check(); await flush(); assert.equal(h.calls.length, 2);
  await h.advance(10_000); assert.equal(h.calls.length, 3);
  let finish;
  h.setHandler(() => new Promise(resolve => { finish = resolve; }));
  await h.advance(10_000); assert.equal(h.calls.length, 4);
  h.tracker.gesture(true); h.tracker.check(); await flush(); assert.equal(h.calls.length, 4);
  h.tracker.stop(); assert.equal(h.calls.at(-1).signal.aborted, true); assert.equal(h.timers.size, 0);
  finish(response(null, 401)); await flush(); assert.equal(h.expired, 0); assert.equal(h.timers.size, 0);
});

test('configured timeout aborts; retries need activity and stay throttled', async () => {
  const h = harness({ info: info({ renew_after_seconds: 0 }) }); await h.start(); await h.advance(10_000);
  h.setHandler((_, signal) => new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted')))));
  h.tracker.gesture(true); await flush(); await h.advance(1999); assert.equal(h.calls.at(-1).signal.aborted, false);
  await h.advance(1); assert.equal(h.calls.at(-1).signal.aborted, true);
  h.tracker.gesture(true); await flush(); assert.equal(h.calls.length, 2);
  h.setVisible(false); await h.advance(100_000); assert.equal(h.calls.length, 2);
});

test('bootstrap failure uses named safety fallback, retries only on later activity', async () => {
  const h = harness({ handler: () => Promise.reject(new Error('offline')) }); await h.start();
  h.tracker.gesture(true); await flush(); assert.equal(h.calls.length, 1);
  await h.advance(BOOTSTRAP_SAFETY_TIMEOUT_MS * 2); assert.equal(h.calls.length, 1);
  h.setHandler(() => Promise.resolve(response())); h.tracker.gesture(true); await flush();
  assert.equal(h.calls.length, 2); assert.equal(h.calls[1].activity, false);
});

test('401 terminates bootstrap or renewal with no request loops', async () => {
  for (const bootstrap of [true, false]) {
    const h = harness({ info: info({ renew_after_seconds: 0 }) });
    if (bootstrap) h.setHandler(() => Promise.resolve(response(null, 401)));
    await h.start();
    if (!bootstrap) { h.setHandler(() => Promise.resolve(response(null, 401))); await h.advance(10_000); h.tracker.gesture(true); await flush(); }
    assert.equal(h.expired, 1); const count = h.calls.length;
    h.tracker.gesture(true); await h.advance(100_000);
    assert.equal(h.calls.length, count); assert.equal(h.timers.size, 0);
  }
});

test('browser integration excludes login and sends only the explicit activity POST', () => {
  const adapter = readFileSync(new URL('./SessionRenewal.tsx', import.meta.url), 'utf8');
  assert.match(adapter, /pathname !== "\/login"/);
  assert.match(adapter, /"X-Echora-Activity": "1"/);
  assert.match(adapter, /event\.isTrusted/);
  assert.match(adapter, /cache: "no-store"/);
  assert.match(adapter, /navigator\.locks\.request/);
  const shell = readFileSync(new URL('../shell/AppShell.tsx', import.meta.url), 'utf8');
  assert.match(shell, /SESSION_EXPIRED_EVENT, expired/);
  assert.match(shell, /generation !== sessionGeneration\(\)/);
});
