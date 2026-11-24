import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./playbackClock.ts', import.meta.url), 'utf8'));
const { smoothPlaybackTime } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('glides between coarse audio updates and never steps backwards', () => {
  let anchor = null;
  const times = [];
  // currentTime only advances every 50 ms while frames arrive every ~16.7 ms.
  for (let frame = 0; frame <= 60; frame++) {
    const now = frame * 1000 / 60;
    const media = Math.floor(now / 50) * .05;
    const result = smoothPlaybackTime(anchor, media, now, true);
    anchor = result.anchor;
    times.push(result.time);
  }
  for (let i = 1; i < times.length; i++) assert.ok(times[i] >= times[i - 1], `frame ${i} went backwards`);
  // Distinct values on most frames: motion is continuous rather than 50 ms steps.
  assert.ok(new Set(times.map(t => t.toFixed(4))).size > 50);
  assert.ok(Math.abs(times.at(-1) - 1) < .06, 'stays close to the real position');
});

test('jumps immediately on a seek and follows the media while paused', () => {
  const playing = smoothPlaybackTime({ media: 10, at: 0, last: 10 }, 42, 16, true);
  assert.equal(playing.time, 42);
  const paused = smoothPlaybackTime(playing.anchor, 41.5, 32, false);
  assert.deepEqual(paused, { time: 41.5, anchor: null });
});
