import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const require = createRequire(import.meta.url);
const helper = stripTypeScriptTypes(readFileSync(new URL('./lyricWords.ts', import.meta.url), 'utf8'));
const lyrics = await import(`data:text/javascript;base64,${Buffer.from(helper).toString('base64')}`);
const { transform, loadBindings } = require('next/dist/build/swc');
await loadBindings();
const { code: compiled } = await transform(readFileSync(new URL('./KaraokeLine.tsx', import.meta.url), 'utf8'), {
  filename: 'KaraokeLine.tsx',
  jsc: { parser: { syntax: 'typescript', tsx: true }, transform: { react: { runtime: 'automatic' } }, target: 'es2022' },
  module: { type: 'commonjs' },
});
const compiledModule = { exports: {} };
new Function('require', 'module', 'exports', compiled)(id => {
  if (id === './lyricWords') return lyrics;
  if (id === './KaraokeLine.module.css') return { line: 'line', paint: 'paint', glowTarget: 'glowTarget', clipDefs: 'clipDefs' };
  return require(id);
}, compiledModule, compiledModule.exports);
const KaraokeLine = compiledModule.exports.default;
const render = texts => {
  const syllables = texts.map((text, index) => ({ text, start_ms: index * 500, end_ms: (index + 1) * 500 }));
  return renderToStaticMarkup(createElement(KaraokeLine, {
    fragments: lyrics.groupSyllablesByWord(syllables).flat(), now: 750, active: true, highlightStyle: 'lava',
  }));
};

test('English syllables share one text node', () => {
  const html = render(['yel', 'low']);
  assert.match(html, /<span>yellow<\/span>/);
  assert.doesNotMatch(html, /<span>yel<\/span>|<span>low<\/span>/);
});

test('Urdu joining is not interrupted at syllable boundaries', () => {
  const html = render(['ار', 'دو']);
  assert.match(html, /<span>اردو<\/span>/);
});

test('mixed-language word order and spaces remain in one native text node', () => {
  const html = render(["I'm ", 'a ', 'pop', 'star, ', 'ناں ', 'ٹوک ', 'مینوں']);
  assert.match(html, /<span>I&#x27;m a popstar, ناں ٹوک مینوں<\/span>/);
  assert.equal((html.match(/dir="auto"/g) || []).length, 1);
  assert.doesNotMatch(html, /<bdi|dir="rtl"|dir="ltr"/);
});

test('paired punctuation can span writing directions without separate boxes', () => {
  assert.match(render(['یہ ', 'میرا ', 'فیشن (', 'میں ', 'کتنا ', 'pas', 'sion)']), /<span>یہ میرا فیشن \(میں کتنا passion\)<\/span>/);
  assert.match(render(['تو ', "'Googles' ", 'کر ', 'لے ', 'یار']), /<span>تو &#x27;Googles&#x27; کر لے یار<\/span>/);
});

test('highlight copy preserves whole text and is hidden from screen readers', () => {
  const html = render(['yel', 'low']);
  assert.match(html, /aria-hidden="true"[^>]*>yellow<\/span>/);
});

test('Japanese kanji carry furigana as native ruby in both text layers', () => {
  const morae = [['か', 0], ['の', 100], ['じょ', 200]].map(([text, at]) => ({ text, start_ms: at, end_ms: at + 100 }));
  const syllables = [
    { text: '「彼女', start_ms: 0, end_ms: 300, ruby: [{ start: 1, end: 3, text: 'かのじょ', morae }] },
    { text: 'は」', start_ms: 300, end_ms: 400 },
  ];
  const fragments = lyrics.groupSyllablesByWord(syllables).flat();
  assert.deepEqual(lyrics.lineRubies(fragments), [{ from: 1, to: 3, text: 'かのじょ', morae }]);
  const html = renderToStaticMarkup(createElement(KaraokeLine, { fragments, now: 150, active: true, highlightStyle: 'lava' }));
  const ruby = '「<ruby>彼女<rt data-ruby="0">かのじょ</rt></ruby>は」';
  assert.equal(html.split(ruby).length - 1, 2);
});
