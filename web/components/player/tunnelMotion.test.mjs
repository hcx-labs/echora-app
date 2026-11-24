import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const source = readFileSync(new URL('./tunnelMotion.ts', import.meta.url), 'utf8');
const { TunnelMotion, gearFor } = await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString('base64')}`);

const input = (overrides = {}) => ({ audible: true, energy: 0.5, anticipation: 0, sides: 6, section: 0, bpm: 120, rate: 1, ...overrides });
// Drive the motion from a beat grid at 60 fps, as the player does: one clock reading per frame.
function play(motion, beats, from, to, overrides = () => ({})) {
  const samples = [];
  for (let t = from; t < to; t += 1 / 60) {
    const clock = clockAt(beats, t);
    motion.receive(clock, t);
    motion.step(1 / 60, t, input(overrides(t)));
    samples.push({ t, travel: motion.travel, gear: motion.gear, rotation: motion.rotation });
  }
  return samples;
}
function clockAt(beats, t) {
  let k = 0;
  while (k < beats.length - 2 && beats[k + 1] <= t) k++;
  return { position: k + (t - beats[k]) / (beats[k + 1] - beats[k]), period: beats[k + 1] - beats[k] };
}

test('rings cross on the beats and follow a tempo change', () => {
  // 120 bpm for 16 beats, then 90 bpm.
  const beats = [];
  for (let i = 0; i < 17; i++) beats.push(i * 0.5);
  for (let i = 1; i < 24; i++) beats.push(8 + i * (2 / 3));
  const samples = play(new TunnelMotion(), beats, 0.2, 20);
  for (const beat of [beats[6], beats[12], beats[20], beats[30]]) {
    const near = samples.reduce((a, b) => (Math.abs(b.t - beat) < Math.abs(a.t - beat) ? b : a));
    assert.ok(Math.abs(near.travel - Math.round(near.travel)) < 0.06, `ring on the beat at ${beat}`);
  }
  const speed = (from, to) => {
    const a = samples.find((s) => s.t >= from), b = samples.find((s) => s.t >= to);
    return (b.travel - a.travel) / (b.t - a.t);
  };
  assert.ok(Math.abs(speed(3, 7) - 2) < 0.05, 'two rings a second at 120 bpm');
  assert.ok(Math.abs(speed(11, 18) - 1.5) < 0.05, 'one and a half at 90 bpm');
});

test('gears change on a bar line and keep rings on the beat', () => {
  const beats = Array.from({ length: 80 }, (_, i) => i * 0.5);
  const motion = new TunnelMotion();
  const samples = play(motion, beats, 0.1, 30, (t) => ({ energy: t < 9.3 ? 0.5 : 0.9 }));
  const change = samples.findIndex((s) => s.gear === 2);
  assert.ok(change > 0);
  // The change waits for the next bar line (beat 20, at 10 s) and does not jump.
  assert.ok(Math.abs(samples[change].t - 10) < 0.05);
  assert.ok(Math.abs(samples[change].travel - samples[change - 1].travel) < 0.1);
  const later = samples.find((s) => s.t >= 15);
  assert.ok(Math.abs(later.travel - Math.round(later.travel)) < 0.06);
});

test('a seek resynchronises instead of racing to catch up', () => {
  const beats = Array.from({ length: 200 }, (_, i) => i * 0.5);
  const motion = new TunnelMotion();
  play(motion, beats, 0.1, 3);
  const samples = play(motion, beats, 60, 62);
  assert.ok(Math.abs(samples[0].travel - Math.round(samples[0].travel) - ((60 / 0.5) % 1)) < 0.1);
  const speed = (samples.at(-1).travel - samples[0].travel) / (samples.at(-1).t - samples[0].t);
  assert.ok(Math.abs(speed - 2) < 0.1);
});

test('rotation turns on downbeats in energetic sections and is continuous across sections', () => {
  const beats = Array.from({ length: 80 }, (_, i) => i * 0.5);
  const motion = new TunnelMotion();
  const samples = play(motion, beats, 0.1, 30, (t) => ({ energy: 0.95, section: t < 16 ? 0 : 1 }));
  const at = (t) => samples.find((s) => s.t >= t).rotation;
  // Bar from 12 s to 14 s: most of the turn in its first beat.
  const firstBeat = at(12.5) - at(12), rest = at(14) - at(12.5);
  assert.ok(firstBeat > rest * 3, `${firstBeat} vs ${rest}`);
  for (let i = 1; i < samples.length; i++) assert.ok(Math.abs(samples[i].rotation - samples[i - 1].rotation) < 0.05);
  // The next section turns the other way.
  assert.ok(at(20) < at(16.1));
});

test('fast songs shift down so rings do not strobe', () => {
  assert.equal(gearFor(0.9, 0, 1, 0.5), 2);
  assert.equal(gearFor(0.9, 0, 1, 0.3), 1);
  assert.equal(gearFor(0.2, 0, 1, 0.5), 0.5);
  assert.equal(gearFor(0.5, 0.6, 1, 0.5), 2);
});
