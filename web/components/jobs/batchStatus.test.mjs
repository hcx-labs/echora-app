import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const source = stripTypeScriptTypes(readFileSync(new URL('./batchStatus.ts', import.meta.url), 'utf8'));
const { batchStatus } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const batch = { status: 'running', phase: 'voice', completed: 12, total: 32, unit: 'tracks', track_count: 32, message: 'Classifying vocals for Song' };
test('running batch identifies its stage, stage count and song message', () => {
  assert.deepEqual(batchStatus(batch), { stage: 'Voice classification', progress: '12 / 32 tracks in this stage', message: 'Classifying vocals for Song' });
});
test('queued batch does not claim to be processing a song', () => {
  assert.deepEqual(batchStatus({ ...batch, status: 'queued' }), { stage: 'Waiting for worker', progress: '32 songs in batch', message: undefined });
});
test('failed batch shows failure rather than stale inference message', () => {
  assert.deepEqual(batchStatus({ ...batch, status: 'failed', error: 'Job execution failed.' }), { stage: 'Failed', progress: '32 songs in batch', message: 'Job execution failed.' });
});
test('model loading progress uses model units, not song count', () => {
  assert.equal(batchStatus({ ...batch, phase: 'models', completed: 0, total: 1, unit: 'models' }).progress, '0 / 1 models in this stage');
});
