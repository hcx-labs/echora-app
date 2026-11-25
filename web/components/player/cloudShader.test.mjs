import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const source = stripTypeScriptTypes(readFileSync(new URL('./cloudShader.ts', import.meta.url), 'utf8'));
const { cloudFragmentShader } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
test('cloud banks have gaps, fine erosion and multiple shadow probes', () => {
  assert.match(cloudFragmentShader, /if\(body<\.005\) return 0\./);
  assert.match(cloudFragmentShader, /max\(0\.,body-erosion\)/);
  assert.match(cloudFragmentShader, /nearProbe/); assert.match(cloudFragmentShader, /farProbe/);
  assert.match(cloudFragmentShader, /i<72/);
});
test('directional lighting keeps artwork pigments and music-driven lightning', () => {
  assert.match(cloudFragmentShader, /secondaryTint/);
  assert.match(cloudFragmentShader, /treble\*\.16/);
  assert.match(cloudFragmentShader, /bolt\*strike/);
});
