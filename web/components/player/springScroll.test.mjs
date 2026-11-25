import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./springScroll.ts', import.meta.url), 'utf8'));
const { springStep, springSettled } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('eases toward the target without overshooting and settles within a second', () => {
  let state = { position: 0, velocity: 0 };
  let settledAt = null;
  for (let frame = 1; frame <= 90; frame++) {
    state = springStep(state, 300, 1 / 60);
    assert.ok(state.position <= 300.5, `overshot on frame ${frame}`);
    if (settledAt === null && springSettled(state, 300)) settledAt = frame;
  }
  assert.ok(settledAt !== null && settledAt <= 60, `settled on frame ${settledAt}`);
});

test('starts gently: the first frame moves only a little', () => {
  const first = springStep({ position: 0, velocity: 0 }, 300, 1 / 60);
  assert.ok(first.position > 0 && first.position < 15);
});

test('retargeting mid-flight keeps velocity instead of restarting', () => {
  let state = { position: 0, velocity: 0 };
  for (let frame = 0; frame < 10; frame++) state = springStep(state, 300, 1 / 60);
  const moving = state.velocity;
  const next = springStep(state, 600, 1 / 60);
  assert.ok(next.velocity >= moving * .9, 'velocity carried over');
});

test('a long dropped frame stays stable', () => {
  const state = springStep({ position: 0, velocity: 0 }, 300, .05);
  assert.ok(Number.isFinite(state.position) && state.position <= 300);
});
