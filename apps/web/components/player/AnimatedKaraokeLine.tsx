"use client";

import { useEffect, useMemo, useRef, type CSSProperties } from "react";
import type { KaraokeClock } from "./KaraokeLine";
import type { LyricFragment } from "./lyricWords";
import styles from "./AnimatedKaraokeLine.module.css";

/** Karaoke styles drawn word by word instead of with a clip-path fill. */
export const ANIMATED_STYLES = ["bloom", "strike"] as const;
export type AnimatedStyle = (typeof ANIMATED_STYLES)[number];

export function isAnimatedStyle(style: string): style is AnimatedStyle {
  return (ANIMATED_STYLES as readonly string[]).includes(style);
}

type Props = {
  fragments: LyricFragment[];
  clock: KaraokeClock;
  active: boolean;
  animation: AnimatedStyle;
};

/** Words: runs of fragments between spaces, kept together so a word never wraps inside. */
function words(fragments: LyricFragment[]) {
  const out: { space: string; parts: { fragment: LyricFragment; index: number }[] }[] = [];
  fragments.forEach((fragment, index) => {
    if (!fragment.text.trim()) {
      out.push({ space: fragment.text, parts: [] });
      return;
    }
    const last = out.at(-1);
    if (last && !last.space) last.parts.push({ fragment, index });
    else out.push({ space: "", parts: [{ fragment, index }] });
  });
  return out;
}

/**
 * A playing line whose words move: "bloom" brings words out of a blur into a soft glow; "strike"
 * punches each word in like a hit in a game, with a slash that charges under it as it is sung.
 * Both start on the syllable's first frame (no easing in), so they never trail the voice.
 * Native text layout still owns wrapping; every frame only flips data attributes and one progress
 * variable on the fragments that change, outside React rendering.
 */
export default function AnimatedKaraokeLine({ fragments, clock, active, animation }: Props) {
  const root = useRef<HTMLSpanElement>(null);
  const grouped = useMemo(() => words(fragments), [fragments]);

  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const nodes = Array.from(element.querySelectorAll<HTMLElement>("[data-fragment]"));
    const last = new Map<HTMLElement, string>();
    const update = () => {
      const now = clock.now();
      for (const node of nodes) {
        const { syllable, text, fragmentIndex } = fragments[Number(node.dataset.fragment)];
        const singing = active && now >= syllable.start_ms && now < syllable.end_ms;
        const state = now >= syllable.end_ms ? "sung" : singing ? "singing" : "waiting";
        let lit = 0;
        if (state === "sung") lit = text.length;
        else if (singing) {
          const elapsed =
            (now - syllable.start_ms) / Math.max(1, syllable.end_ms - syllable.start_ms);
          lit = Math.max(
            0,
            Math.min(text.length, Math.ceil(elapsed * syllable.text.length - fragmentIndex)),
          );
        }
        const key = `${state}:${lit}`;
        if (last.get(node) === key) continue;
        last.set(node, key);
        node.dataset.state = state;
        if (singing) node.setAttribute("data-lyric-singing", "true");
        else node.removeAttribute("data-lyric-singing");
        node.style.setProperty("--p", String(text.length ? lit / text.length : 0));
        // Letters light up as the singing reaches them.
        Array.from(node.children).forEach((letter, index) => {
          (letter as HTMLElement).dataset.on = index < lit ? "1" : "";
        });
      }
    };
    update();
    return clock.subscribe(update);
  }, [clock, fragments, active, grouped]);

  return (
    <span ref={root} dir="auto" className={styles.line} data-animation={animation}>
      {grouped.map((word, wordIndex) =>
        word.space ? (
          word.space
        ) : (
          <span key={wordIndex} className={styles.word}>
            {word.parts.map(({ fragment, index }) => (
              <span
                key={index}
                className={styles.fragment}
                data-fragment={index}
                data-state="waiting"
              >
                {Array.from(fragment.text).map((letter, letterIndex) => (
                  <span
                    key={letterIndex}
                    className={styles.letter}
                    style={{ "--i": letterIndex } as CSSProperties}
                  >
                    {letter}
                  </span>
                ))}
              </span>
            ))}
          </span>
        ),
      )}
    </span>
  );
}
