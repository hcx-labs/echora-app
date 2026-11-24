import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import test from 'node:test';

const components = fileURLToPath(new URL('../components/', import.meta.url));
function styles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? styles(path) : entry.name.endsWith('.module.css') ? [path] : [];
  });
}

test('component panels use shared surfaces instead of fixed dark colors', () => {
  for (const path of styles(components)) {
    const source = readFileSync(path, 'utf8');
    for (const match of source.matchAll(/background(?:-color)?\s*:\s*#([\da-f]{6}|[\da-f]{3})\s*[;}]/gi)) {
      const hex = match[1].length === 3 ? [...match[1]].map(c => c + c).join('') : match[1];
      const rgb = [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
      const neutralDark = Math.max(...rgb) <= 58 && Math.max(...rgb) - Math.min(...rgb) <= 25;
      assert.equal(neutralDark, false, `${path} has a fixed panel background: ${match[0]}`);
    }
    assert.ok(!source.includes('--art-panel,'), `${path} bypasses shared surfaces`);
    assert.doesNotMatch(source, /--[\w-]*surface[\w-]*\s*:\s*#/, `${path} has a fixed surface alias`);
  }
});

test('the fullscreen player stays translucent so the backdrop remains visible', () => {
  const player = readFileSync(join(components, 'player/FullscreenPlayer.module.css'), 'utf8');
  const tokens = readFileSync(new URL('./theme.css', import.meta.url), 'utf8');
  const surface = player.match(/\.player\[data-slot="dialog-content"\] \{([^}]*)\}/)?.[1] ?? '';
  assert.match(surface, /background: linear-gradient\(/);
  for (const stop of surface.matchAll(/rgb\(0 0 0 \/ ([\d.]+)\)/g)) assert.ok(Number(stop[1]) < 1, 'scrim stops must stay see-through');
  assert.match(tokens, /--surface-shell:\s*rgb\(var\(--surface-rgb\)\s*\/\s*\.14\)/);
});

// The app accent follows the playing artwork, so status colours must never use it.
test('status notices and toasts use fixed colours, not the artwork accent', () => {
  const notice = readFileSync(join(components, 'ui/notice.tsx'), 'utf8');
  const toaster = readFileSync(join(components, 'ui/sonner.tsx'), 'utf8');
  for (const [tone, token] of [['success', 'success'], ['warning', 'warning'], ['error', 'destructive']]) {
    const noticeTone = notice.match(new RegExp(`${tone}: \\{ Icon: \\w+, className: "([^"]*)"`))?.[1] ?? '';
    assert.match(noticeTone, new RegExp(`border-${token}`), `${tone} notice border`);
    assert.doesNotMatch(noticeTone, /primary|accent/, `${tone} notice uses the accent`);
    assert.match(toaster, new RegExp(`${tone}: <\\w+ className="[^"]*text-${token}`), `${tone} toast icon`);
    assert.match(toaster, new RegExp(`${tone}: "[^"]*border-l-${token}`), `${tone} toast edge`);
  }
});

test('palette publication changes only the shared surface hue', () => {
  const provider = readFileSync(join(components, 'player/PlayerProvider.tsx'), 'utf8');
  assert.match(provider, /setProperty\("--surface-rgb", panelColorFromPalette/);
  assert.match(provider, /removeProperty\("--surface-rgb"\)/);
  assert.doesNotMatch(provider, /setProperty\("--(?:glass|art-panel)"/);
});
