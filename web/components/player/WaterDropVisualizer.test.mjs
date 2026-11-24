import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
const source = readFileSync(new URL('./WaterDropVisualizer.tsx', import.meta.url), 'utf8');
test('drop impacts remain simulated without rendering droplets', () => {
  assert.match(source, /drop\.velocity \+= controls\.gravity/);
  assert.match(source, /surface\.impact\(\s*drop\.x/);
  assert.doesNotMatch(source, /InstancedMesh|SphereGeometry/);
});
test('water uses up to five extracted album colors with smooth transitions', () => {
  assert.match(source, /palette\.trails\?\.length \? palette\.trails : palette\.waves/);
  assert.match(source, /Math\.min\(5, extracted\.length\)/);
  assert.match(source, /uniform vec3 colorA, colorB, colorC, colorD, colorE/);
  assert.match(source, /\.value\.lerp\(colors\[index\], response\)/);
});
