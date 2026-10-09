import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./lyricWords.ts', import.meta.url), 'utf8'));
const { groupSyllablesByWord, fragmentProgress, lyricWordIsRtl, splitVoices } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const syllable = text => ({ text, start_ms: 1000, end_ms: 2000 });
const wordText = words => words.map(word => word.map(fragment => fragment.text).join(''));

for (const text of [
  '「危ないですから離れてください」そのセリフが集合の合図なのにな',
  '这是没有空格的一行歌词需要自动换行',
  'นี่คือเนื้อเพลงภาษาไทยที่ไม่มีช่องว่าง',
]) {
  test(`finds wrap points without spaces: ${text}`, () => {
    const original = syllable(text);
    const words = groupSyllablesByWord([original]);
    assert.ok(words.length > 2);
    assert.equal(wordText(words).join(''), text);
    assert.ok(words.flat().every(fragment => fragment.syllable === original));
  });
}

test('keeps Latin words intact across original syllable boundaries', () => {
  const syllables = ['Hel', 'lo ', 'beau', 'ti', 'ful ', 'world'].map(syllable);
  const words = groupSyllablesByWord(syllables);
  assert.deepEqual(wordText(words), ['Hello ', 'beautiful ', 'world']);
  assert.deepEqual(words[1].map(fragment => fragment.index), [2, 3, 4, 4]);
  for (const fragment of words.flat()) {
    assert.equal(fragment.text, fragment.syllable.text.slice(fragment.fragmentIndex, fragment.fragmentIndex + fragment.text.length));
  }
});

test('keeps Japanese brackets with their adjacent word', () => {
  const words = wordText(groupSyllablesByWord([syllable('「離れてください」その合図')]));
  assert.ok(words[0].startsWith('「'));
  assert.ok(words.every(word => !word.startsWith('」') && !word.endsWith('「')));
});

test('preserves RTL, combining marks, emoji, whitespace, and empty syllables', () => {
  for (const text of ['مرحبا بالعالم', 'שָׁלוֹם עולם', 'cafe\u0301 👨‍👩‍👧‍👦', 'a  b\n c', '', '   ']) {
    assert.equal(wordText(groupSyllablesByWord([syllable(''), syllable(text), syllable('')])).join(''), text);
  }
  assert.deepEqual(groupSyllablesByWord([]), []);
});

test('lava fill continues across word fragments rather than restarting', () => {
  const original = syllable('hello world');
  const fragments = groupSyllablesByWord([original]).flat();
  assert.equal(fragmentProgress(1500, fragments[0]), 100);
  assert.equal(fragmentProgress(1500, fragments.at(-1)), 0);
  assert.equal(fragmentProgress(1750, fragments.at(-1)), 45);
  assert.equal(fragmentProgress(2500, fragments.at(-1)), 100);
});


test('mixed Urdu and English retains complete words across syllable boundaries', () => {
  const syllables = ['او ', 'yel', 'low ', 'کو ', 'ار', 'دو ', 'میں ', 'بو', 'لتے ', 'ہیں'].map(syllable);
  const words = wordText(groupSyllablesByWord(syllables));
  assert.deepEqual(words, ['او ', 'yellow ', 'کو ', 'اردو ', 'میں ', 'بولتے ', 'ہیں']);
  assert.equal(lyricWordIsRtl(words[0]), true);
  assert.equal(lyricWordIsRtl(words[1]), false);
  assert.equal(lyricWordIsRtl(words[3]), true);
});

test('word direction follows its first strong letter, not punctuation or surrounding script', () => {
  assert.equal(lyricWordIsRtl('(yellow)'), false);
  assert.equal(lyricWordIsRtl('123اردو'), true);
  assert.equal(lyricWordIsRtl('「日本語」'), false);
  assert.equal(lyricWordIsRtl('שלום'), true);
  assert.equal(lyricWordIsRtl('123'), false);
});


test('brackets and quotes survive fragment segmentation unchanged', () => {
  for (const text of ["تو 'Googles' کر لے یار", 'یہ میرا فیشن (میں کتنا passion)', "I'm a popstar, ناں ٹوک مینوں"]) {
    assert.equal(wordText(groupSyllablesByWord([syllable(text)])).join(''), text);
  }
});

test('backing vocals become their own line without the parentheses', () => {
  const at = (text, start, voice) => ({ text, start_ms: start, end_ms: start + 100, ...(voice ? { voice } : {}) });
  const line = [at('Lone', 0), at('ly (', 100), at('so ', 50, 2), at('lone', 150, 2), at('ly)', 250, 2)];
  const { lead, backing } = splitVoices(line);
  assert.deepEqual(lead.map(s => s.text), ['Lone', 'ly']);
  assert.deepEqual(backing.map(s => s.text), ['so ', 'lone', 'ly']);
  assert.equal(backing[0].start_ms, 50);  // timing is kept, so both lines can fill at once
  const plain = [at('No ', 0), at('backing', 100)];
  assert.equal(splitVoices(plain).lead, plain);  // untouched when there is no second voice
  assert.equal(splitVoices(plain).backing.length, 0);
});
