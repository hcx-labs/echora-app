import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runtimeConfigModule } from './runtimeConfig.test-helper.mjs';
const { runtimeConfig, applyRuntimeConfig } = await import(runtimeConfigModule);
const defaults = JSON.parse(readFileSync(new URL('./runtimeConfig.defaults.json', import.meta.url), 'utf8'));

test('generated defaults work before loading; valid settings replace only allowed values', () => {
  assert.deepEqual(runtimeConfig(), defaults);
  assert.equal(applyRuntimeConfig({ ...defaults, job_poll_ms: 4200, secret: 'never expose' }), true);
  assert.equal(runtimeConfig().job_poll_ms, 4200);
  assert.equal('secret' in runtimeConfig(), false);
  applyRuntimeConfig(defaults);
});

test('malformed config preserves previous values instead of creating tight timer loops', () => {
  for (const value of [null, {}, { ...defaults, job_poll_ms: 0 }, { ...defaults, job_poll_ms: -1 },
    { ...defaults, job_poll_ms: 0.5 }, { ...defaults, job_poll_ms: '1200' },
    { ...defaults, job_poll_ms: 2 ** 31 }, { ...defaults, job_poll_ms: NaN }]) {
    assert.equal(applyRuntimeConfig(value), false);
    assert.deepEqual(runtimeConfig(), defaults);
  }
});
