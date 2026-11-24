import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = readFileSync(new URL('./FullscreenPlayer.tsx', import.meta.url), 'utf8');
test('synced lyrics do not mount a glow canvas or mark whole lines as singing', () => {
  assert.match(source, /karaokeMode && karaokeAvailable && \(?\s*<LyricsGlow/);
  assert.doesNotMatch(source, /data-lyric-singing/);
});
