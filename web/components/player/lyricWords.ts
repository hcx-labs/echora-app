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

const BRACKETS = /[()（）]/g;

/** Which of these texts are written inside round brackets (most of their letters are).
 *  A bracket opened in one text stays open in the next: "(so ", "lone", "ly) ". */
function bracketed(texts: string[]): boolean[] {
  let depth = 0;
  return texts.map((text) => {
    let inside = 0,
      outside = 0;
    for (const character of text) {
      if (character === "(" || character === "（") depth++;
      else if (character === ")" || character === "）") depth = Math.max(0, depth - 1);
      else if (/[\p{L}\p{N}]/u.test(character)) {
        if (depth > 0) inside++;
        else outside++;
      }
    }
    return inside > 0 && inside >= outside;
  });
}

const PUNCTUATION = ",.!?;:、。！？，";
const OPENS = /[(（]/;
const CLOSES = /[)）]/;

/** Append a syllable with its brackets dropped, keeping one space between words and none
 *  before punctuation ("far away (…), is" becomes "far away, is"). */
function addTo(part: KaraokeSyllable[], syllable: KaraokeSyllable, raw: string) {
  let text = raw.replace(BRACKETS, "").replace(/\s+/g, " ");
  const last = part.at(-1);
  if (!last || last.text.endsWith(" ")) text = text.trimStart();
  if (last && PUNCTUATION.includes(text[0] ?? "") && last.text.endsWith(" "))
    part[part.length - 1] = { ...last, text: last.text.trimEnd() };
  if (!text.trim()) {
    // A bracket alone or a gap: keep the word break, drop the syllable.
    const previous = part.at(-1);
    if (text && previous && !previous.text.endsWith(" "))
      part[part.length - 1] = { ...previous, text: `${previous.text} ` };
    return;
  }
  part.push({ ...syllable, text });
}

/** Close one backing part before the next: "hey" becomes "hey, ". */
function endPart(part: KaraokeSyllable[]) {
  const last = part[part.length - 1];
  let text = last.text.trimEnd();
  if (text && !PUNCTUATION.includes(text.at(-1)!)) text += ",";
  part[part.length - 1] = { ...last, text: `${text} ` };
}

/** Backing vocals shown as their own, smaller line under the lead: syllables marked
 *  ``voice: 2`` and everything written in round brackets, which is always a backing vocal
 *  (also in lyrics stored before they were marked). The brackets themselves are dropped: the
 *  separate line says it already. Each bracketed part is one backing part, and parts are joined
 *  with commas: "Hello (hey) I am (hey)" is lead "Hello I am", backing "hey, hey". A line that
 *  is bracketed as a whole has no lead part. Same rule as backing_vocals.separate_voices. */
export function splitVoices(syllables: KaraokeSyllable[]): {
  lead: KaraokeSyllable[];
  backing: KaraokeSyllable[];
} {
  const inBrackets = bracketed(syllables.map((syllable) => syllable.text));
  const isBacking = syllables.map((syllable, index) => syllable.voice === 2 || inBrackets[index]);
  if (!isBacking.some(Boolean)) return { lead: syllables, backing: [] };
  const lead: KaraokeSyllable[] = [];
  const backing: KaraokeSyllable[] = [];
  let newPart = false;
  syllables.forEach((syllable, index) => {
    if (isBacking[index]) {
      if (backing.length && (newPart || OPENS.test(syllable.text))) endPart(backing);
      newPart = CLOSES.test(syllable.text);
      addTo(backing, syllable, syllable.text);
    } else {
      // Lead words between two backing runs split them into separate parts.
      newPart = newPart || backing.length > 0;
      addTo(lead, syllable, syllable.text);
    }
  });
  for (const part of [lead, backing])
    if (part.length) part[part.length - 1] = { ...part.at(-1)!, text: part.at(-1)!.text.trimEnd() };
  return { lead, backing };
}

/** The same split for a line without syllables: its text outside and inside round brackets. */
export function splitTextVoices(text: string): { lead: string; backing: string } {
  if (!OPENS.test(text)) return { lead: text, backing: "" };
  const pieces = text.split(/([(（][^()（）]*[)）])/);
  const syllables = pieces
    .map((piece, index) => ({
      text: piece,
      start_ms: 0,
      end_ms: 0,
      ...(index % 2 ? { voice: 2 } : {}),
    }))
    .filter((piece) => piece.text);
  const { lead, backing } = splitVoices(syllables as KaraokeSyllable[]);
  const join = (part: KaraokeSyllable[]) => part.map((syllable) => syllable.text).join("");
  return { lead: join(lead), backing: join(backing) };
}

type TimedLine = {
  start_ms: number | null;
  end_ms?: number;
  text: string;
  syllables?: KaraokeSyllable[];
};

/** A line whose sung words are all backing vocals ("(One, two, three)"). */
export function isBackingLine(line: TimedLine): boolean {
  if (line.syllables?.length) {
    const { lead, backing } = splitVoices(line.syllables);
    return !lead.length && backing.length > 0;
  }
  const { lead, backing } = splitTextVoices(line.text);
  return !lead && Boolean(backing);
}

function lineEnd(line: TimedLine): number {
  return Math.max(
    Number(line.end_ms) || 0,
    ...(line.syllables || []).map((syllable) => syllable.end_ms),
  );
}

/** How far a backing line must reach back into the lead line before it to be sung with it. */
const FOLD_OVERLAP_MS = 150;

/** A backing line sung over the lead line before it joins that line as its backing part, so the
 *  two are shown (and highlighted) together. A backing line sung on its own keeps its own slot. */
export function foldBackingLines<T extends TimedLine>(lines: T[]): T[] {
  const folded: T[] = [];
  for (const line of lines) {
    const previous = folded.at(-1);
    if (
      previous?.syllables?.length &&
      line.syllables?.length &&
      Number.isFinite(line.start_ms) &&
      isBackingLine(line) &&
      !isBackingLine(previous) &&
      Number(line.start_ms) < lineEnd(previous) - FOLD_OVERLAP_MS
    ) {
      const syllables = [...previous.syllables];
      const last = syllables[syllables.length - 1];
      // Keep a word break between the previous line's words and the joined ones.
      if (!/\s$/u.test(last.text))
        syllables[syllables.length - 1] = { ...last, text: `${last.text} ` };
      folded[folded.length - 1] = {
        ...previous,
        end_ms: Math.max(lineEnd(previous), lineEnd(line)),
        syllables: [...syllables, ...line.syllables.map((syllable) => ({ ...syllable, voice: 2 }))],
      };
      continue;
    }
    folded.push(line);
  }
  return folded;
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
