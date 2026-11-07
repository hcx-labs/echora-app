/** One timed mora of a reading: the kana and when it is sung. */
export type RubyMora = { text: string; start_ms: number; end_ms: number };
/** Furigana over `text.slice(start, end)` of a syllable (Japanese kanji). */
export type RubySegment = { start: number; end: number; text: string; morae: RubyMora[] };
export type KaraokeSyllable = {
  start_ms: number;
  end_ms: number;
  text: string;
  voice?: number;
  ruby?: RubySegment[];
};
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

/** Backing-vocal syllables (``voice: 2``) shown as their own line under the lead.
 *  The parentheses that marked the backing part are dropped: the separate line says it already.
 *  A line without backing syllables comes back unchanged, as the lead alone. */
export function splitVoices(syllables: KaraokeSyllable[]): {
  lead: KaraokeSyllable[];
  backing: KaraokeSyllable[];
} {
  if (!syllables.some((syllable) => syllable.voice === 2)) return { lead: syllables, backing: [] };
  const clean = (part: KaraokeSyllable[]) => {
    const cleaned = part
      .map((syllable) => ({ ...syllable, text: syllable.text.replace(/[()]/g, "") }))
      .filter((syllable) => syllable.text.trim());
    if (cleaned.length)
      cleaned[cleaned.length - 1] = { ...cleaned.at(-1)!, text: cleaned.at(-1)!.text.trimEnd() };
    return cleaned;
  };
  return {
    lead: clean(syllables.filter((syllable) => syllable.voice !== 2)),
    backing: clean(syllables.filter((syllable) => syllable.voice === 2)),
  };
}

/** A reading placed on a line: furigana over `text.slice(from, to)` of the joined fragments. */
export type LineRuby = { from: number; to: number; text: string; morae: RubyMora[] };

/** Readings of a line's fragments, as offsets into their joined text (whole segments only). */
export function lineRubies(fragments: LyricFragment[]): LineRuby[] {
  const rubies: LineRuby[] = [];
  let offset = 0;
  for (const fragment of fragments) {
    const first = fragment.fragmentIndex;
    const last = first + fragment.text.length;
    for (const segment of fragment.syllable.ruby || []) {
      if (segment.start >= first && segment.end <= last && segment.end > segment.start)
        rubies.push({
          from: offset + segment.start - first,
          to: offset + segment.end - first,
          text: segment.text,
          morae: segment.morae,
        });
    }
    offset += fragment.text.length;
  }
  return rubies;
}
