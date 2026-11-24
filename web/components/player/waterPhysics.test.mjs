import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const source = stripTypeScriptTypes(readFileSync(new URL('./waterPhysics.ts', import.meta.url), 'utf8'));
const { WaterSurface, waterControls } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const preferences = { bassReactivity: 1, vocalReactivity: 1, trebleReactivity: 1 };
const frame = { active: true, bass: 0, mid: 0, treble: 0, bassAttack: 0, trebleAttack: 0, bpm: 100 };
const energy = surface => surface.height.reduce((total, h, i) => total + h * h + surface.velocity[i] ** 2, 0);

test('music controls drop rate, velocity, radius, tension, damping and impact', () => {
  const quiet = waterControls(frame, preferences);
  const bass = waterControls({ ...frame, bass: 1, bassAttack: .2 }, preferences);
  const mid = waterControls({ ...frame, mid: 1 }, preferences);
  const treble = waterControls({ ...frame, treble: 1 }, preferences);
  assert.equal(quiet.rate, 0);
  assert.ok(bass.rate > 0 && bass.radius > quiet.radius && bass.impact > quiet.impact);
  assert.ok(mid.fallSpeed > quiet.fallSpeed && mid.tension > quiet.tension);
  assert.ok(treble.tension > quiet.tension && treble.damping < quiet.damping);
  assert.ok(waterControls({ ...frame, bass: 1, bpm: 180 }, preferences).rate > bass.rate);
  const muted = waterControls({ ...frame, bass: 1, mid: 1, treble: 1, bassAttack: 1 }, { bassReactivity: 0, vocalReactivity: 0, trebleReactivity: 0 });
  assert.equal(muted.rate, 0); assert.equal(muted.attack, 0);
  assert.equal(waterControls({ ...frame, active: false, bass: 1 }, preferences).rate, 0);
});

test('impacts propagate beyond their initial radius and settle', () => {
  const surface = new WaterSurface(64, 40);
  surface.impact(.5, .5, 2, .8);
  for (let i = 0; i < 60; i++) surface.advance(1 / 60, .3, .984);
  assert.ok(Math.abs(surface.height[20 * 64 + 45]) > .00001);
  const before = energy(surface);
  for (let i = 0; i < 600; i++) surface.advance(1 / 60, .3, .984);
  assert.ok(energy(surface) < before * .001);
  assert.ok(surface.height.every(Number.isFinite));
});

test('wave equation is linear so simultaneous ripples interfere', () => {
  const a = new WaterSurface(32, 24), b = new WaterSurface(32, 24), combined = new WaterSurface(32, 24);
  a.impact(.35, .5, 2, .2); b.impact(.65, .5, 2, .2);
  combined.impact(.35, .5, 2, .2); combined.impact(.65, .5, 2, .2);
  for (let i = 0; i < 30; i++) for (const surface of [a, b, combined]) surface.advance(1 / 60, .25, .985);
  combined.height.forEach((h, i) => assert.ok(Math.abs(h - a.height[i] - b.height[i]) < .00001));
});

test('fixed-step results match at 30 and 60 FPS', () => {
  const a = new WaterSurface(32, 24), b = new WaterSurface(32, 24);
  a.impact(.5, .5, 2, .5); b.impact(.5, .5, 2, .5);
  for (let i = 0; i < 60; i++) a.advance(1 / 60, .3, .984);
  for (let i = 0; i < 30; i++) b.advance(1 / 30, .3, .984);
  a.height.forEach((h, i) => assert.ok(Math.abs(h - b.height[i]) < .000001));
});

test('bounded simulation handles repeated impacts and invalid input', () => {
  const surface = new WaterSurface(32, 24);
  surface.impact(NaN, .5, 2, 1); surface.advance(NaN, .3, .99);
  for (let i = 0; i < 300; i++) {
    if (i % 10 === 0) surface.impact(.5, .5, 8, 1.5);
    surface.advance(1 / 30, .45, .995);
  }
  assert.ok(surface.height.every(h => Number.isFinite(h) && Math.abs(h) <= 2));
  surface.clear(); assert.equal(energy(surface), 0);
});
