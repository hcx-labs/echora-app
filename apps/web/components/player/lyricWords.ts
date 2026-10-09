export type KaraokeSyllable = { start_ms: number; end_ms: number; text: string };
export type LyricFragment = {
  syllable: KaraokeSyllable;
  index: number;
  text: string;
  fragmentIndex: number;
};

const segmenter = new Intl.Segmenter(undefined, { granularity: "word" });

export function lyricWordIsRtl(text: string) {
  const firstLetter = /\p{L}/u.exec(text)?.[0];
  return !!firstLetter && /[\p{Script=Arabic}\p{Script=Hebrew}]/u.test(firstLetter);
}

/** Find language-aware wrap points without losing source syllables or their timestamps. */
export function groupSyllablesByWord(syllables: KaraokeSyllable[]): LyricFragment[][] {
  const text = syllables.map((syllable) => syllable.text).join("");
  const words: string[] = [];
  let prefix = "";
  for (const part of segmenter.segment(text)) {
    if (part.isWordLike) {
      words.push(prefix + part.segment);
      prefix = "";
    } else if (
      /^[\p{Ps}\p{Pi}]+$/u.test(part.segment) ||
      (/^["']$/u.test(part.segment) && /\s$/u.test(words.at(-1) || "")) ||
      prefix ||
      !words.length
    ) {
      // Opening brackets/quotes belong to the next word, closing punctuation to the last.
      prefix += part.segment;
    } else {
      words[words.length - 1] += part.segment;
    }
  }
  if (prefix) words.push(prefix);

  let index = 0,
    offset = 0;
  return words.map((word) => {
    const fragments: LyricFragment[] = [];
    let remaining = word.length;
    while (remaining > 0 && index < syllables.length) {
      const syllable = syllables[index];
      const length = Math.min(remaining, syllable.text.length - offset);
      // Keep spaces out of the karaoke mask, including spaces attached to a word.
      for (const text of syllable.text
        .slice(offset, offset + length)
        .split(/(\s+)/u)
        .filter(Boolean)) {
        fragments.push({ syllable, index, text, fragmentIndex: offset });
        offset += text.length;
      }
      remaining -= length;
      if (offset === syllable.text.length) {
        index++;
        offset = 0;
      }
    }
    return fragments;
  });
}

/** Each fragment paints its part of the original syllable rather than restarting the fill. */
export function fragmentProgress(now: number, { syllable, text, fragmentIndex }: LyricFragment) {
  const elapsed = (now - syllable.start_ms) / Math.max(1, syllable.end_ms - syllable.start_ms);
  return Math.min(
    100,
    Math.max(
      0,
      ((elapsed * syllable.text.length - fragmentIndex) / Math.max(1, text.length)) * 100,
    ),
  );
}
