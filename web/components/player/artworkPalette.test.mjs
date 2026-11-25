// Run: node --test web/components/player/artworkPalette.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { stripTypeScriptTypes } from 'node:module';

// Node 22's built-in type stripping; no test runner or browser dependencies.
const source = readFileSync(new URL('./artworkPalette.ts', import.meta.url), 'utf8');
const outputText = stripTypeScriptTypes(source);
const { paletteFromPixels, panelColorFromPalette } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const pixels = (...regions) => new Uint8ClampedArray(regions.flatMap(([rgb, count, alpha = 255]) => Array.from({ length: count }, () => [...rgb, alpha]).flat()));
const lum = rgb => rgb.map(n => n / 255).map(n => n <= .04045 ? n / 12.92 : ((n + .055) / 1.055) ** 2.4).reduce((sum, n, i) => sum + n * [.2126, .7152, .0722][i], 0);
const contrast = (a, b) => (Math.max(lum(a), lum(b)) + .05) / (Math.min(lum(a), lum(b)) + .05);
const spread = rgb => Math.max(...rgb) - Math.min(...rgb);

test('panel tones follow artwork hue and keep readable contrast', () => {
  for (const color of [[230, 30, 50], [20, 200, 80], [30, 70, 240], [210, 160, 20], [128, 128, 128]]) {
    const palette = paletteFromPixels(pixels([color, 100]));
    const panel = panelColorFromPalette(palette);
    assert.ok(contrast([242, 244, 241], panel) >= 12);
    assert.ok(contrast([137, 150, 145], panel) >= 4.5);
    assert.ok(contrast(palette.accent, panel) >= 4.5);
    assert.ok(Math.max(...panel) <= 20, `Panel must stay near-black: ${panel}`);
    assert.ok(lum(panel) < .006, `Panel is too bright: ${panel}`);
    if (spread(color) === 0) assert.ok(spread(panel) <= 1);
    else assert.equal(panel.indexOf(Math.max(...panel)), color.indexOf(Math.max(...color)));
  }
});

function valid(palette) {
  for (const rgb of [palette.accent, palette.background]) for (const n of rgb) assert.ok(Number.isInteger(n) && n >= 0 && n <= 255);
  for (const rgb of [...palette.waves, ...palette.trails]) for (const n of rgb) assert.ok(Number.isFinite(n) && n >= 0 && n <= 1);
  assert.ok(contrast(palette.accent, [7, 16, 15]) >= 4.5);
  assert.ok(contrast(palette.accent, palette.background) >= 4.5);
  assert.ok(contrast(palette.accent.map(n => n * .88), [22, 48, 45]) >= 4.5);
}

test('meaningful vibrant accent wins over neutral majority', () => {
  const p = paletteFromPixels(pixels([[130, 130, 130], 900], [[220, 35, 65], 100]));
  valid(p);
  assert.ok(p.accent[0] > p.accent[1] + 80);
  assert.ok(p.accent[0] > p.accent[2] + 60);
});

test('tiny saturated noise cannot become accent or a wave', () => {
  const p = paletteFromPixels(pixels([[100, 100, 100], 995], [[255, 0, 200], 5]));
  valid(p);
  assert.ok(spread(p.accent) <= 1);
  p.waves.forEach(c => assert.ok(spread(c) <= 1 / 255));
});

test('nearby source shades pool their support across RGB boundaries', () => {
  const regions = Array.from({ length: 10 }, (_, i) => [[210 + i, 30 + i, 50 + i], 10]);
  const p = paletteFromPixels(pixels([[120, 120, 120], 900], ...regions));
  valid(p);
  assert.ok(p.accent[0] > p.accent[1] + 80);
});

test('saturated dark colors brighten without losing their hue family', () => {
  for (const [color, channel] of [[[80, 0, 0], 0], [[0, 60, 0], 1], [[0, 0, 100], 2]]) {
    const p = paletteFromPixels(pixels([color, 100]));
    valid(p);
    assert.equal(p.accent.indexOf(Math.max(...p.accent)), channel);
    assert.ok(spread(p.accent) > 90);
  }
});

test('gray, black and white remain achromatic with tonal waves', () => {
  for (const value of [0, 100, 255]) {
    const p = paletteFromPixels(pixels([[value, value, value], 100]));
    valid(p);
    assert.ok(spread(p.accent) <= 1);
    p.waves.forEach(c => assert.ok(spread(c) <= 1 / 255));
    assert.notDeepEqual(p.waves[1], p.waves[2]);
  }
});

test('transparent colors ignored, alpha cutoff retained, empty input throws', () => {
  assert.throws(() => paletteFromPixels(new Uint8ClampedArray()), /no usable colors/);
  assert.throws(() => paletteFromPixels(pixels([[255, 0, 0], 100, 0])), /no usable colors/);
  const opaque = pixels([[30, 160, 100], 100, 180]);
  assert.deepEqual(paletteFromPixels(pixels([[255, 0, 0], 1000, 179], [[30, 160, 100], 100, 180])), paletteFromPixels(opaque));
  valid(paletteFromPixels(opaque));
});

test('two and three meaningful colors produce separated source waves', () => {
  for (const regions of [
    [[[240, 30, 40], 600], [[20, 150, 230], 400]],
    [[[240, 30, 40], 400], [[20, 150, 230], 300], [[30, 200, 50], 300]],
  ]) {
    const p = paletteFromPixels(pixels(...regions));
    valid(p);
    for (let i = 0; i < regions.length; i++) for (let j = i + 1; j < regions.length; j++) {
      assert.ok(Math.hypot(...p.waves[i].map((n, k) => n - p.waves[j][k])) > .4);
    }
  }
});

test('deterministic, order-independent, non-mutating and bounded across RGB gamut', () => {
  const input = pixels([[255, 30, 20], 50], [[20, 80, 210], 50]);
  const copy = input.slice();
  assert.deepEqual(paletteFromPixels(input), paletteFromPixels(input));
  assert.deepEqual(input, copy);
  assert.deepEqual(paletteFromPixels(input), paletteFromPixels(pixels([[20, 80, 210], 50], [[255, 30, 20], 50])));
  for (const r of [0, 64, 128, 192, 255]) for (const g of [0, 64, 128, 192, 255]) for (const b of [0, 64, 128, 192, 255]) valid(paletteFromPixels(pixels([[r, g, b], 4])));
});


test('trail palette retains five distinct supported artwork colors', () => {
  const regions = [[[240, 30, 40], 200], [[20, 150, 230], 200], [[30, 200, 50], 200], [[240, 210, 20], 200], [[190, 30, 220], 200]];
  const p = paletteFromPixels(pixels(...regions));
  valid(p);
  assert.equal(p.trails.length, 5);
  assert.equal(new Set(p.trails.map(c => c.join(','))).size, 5);
  assert.equal(p.waves.length, 3);
  assert.deepEqual(p.trails, paletteFromPixels(pixels(...regions.reverse())).trails);
});

test('trail palette does not invent colors for simple covers or amplify tiny specks', () => {
  const p = paletteFromPixels(pixels([[100, 100, 100], 995], [[255, 0, 200], 5]));
  assert.equal(p.trails.length, 1);
  assert.ok(spread(p.trails[0]) <= 1 / 255);
  const two = paletteFromPixels(pixels([[240, 30, 40], 500], [[20, 150, 230], 500]));
  assert.equal(two.trails.length, 2);
});
