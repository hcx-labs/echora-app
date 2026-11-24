import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./motionArtwork.ts', import.meta.url), 'utf8'));
const { resolveArtworkStyle, motionArtworkPath, motionVideoPath } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('a saved artwork style wins; otherwise reduced motion picks the still cover', () => {
  assert.equal(resolveArtworkStyle(null, false), 'motion');
  assert.equal(resolveArtworkStyle(null, true), 'still');
  assert.equal(resolveArtworkStyle('motion', true), 'motion');
  assert.equal(resolveArtworkStyle('still', false), 'still');
  assert.equal(resolveArtworkStyle('sparkles', false), 'motion');
});

test('looks up motion artwork per track', () => {
  assert.equal(motionArtworkPath('a/b'), '/analysis/library/tracks/a%2Fb/motion-artwork');
});

test('accepts only an available loop on the motion artwork route', () => {
  const url = '/motion-artwork/0f8fad5b-d9cb-469f-a165-70867728950e.mp4';
  assert.equal(motionVideoPath({ available: true, url }), url);
  assert.equal(motionVideoPath({ available: false, url }), null);
  assert.equal(motionVideoPath({ available: true }), null);
  assert.equal(motionVideoPath({ available: true, url: 'https://elsewhere.example/x.mp4' }), null);
  assert.equal(motionVideoPath({ available: true, url: '/motion-artwork/../secret.mp4' }), null);
  assert.equal(motionVideoPath(null), null);
});
