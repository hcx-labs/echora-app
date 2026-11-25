import type { LyricTranslation } from "./PlayerProvider";
import styles from "./LyricTranslationLine.module.css";

// Match source text, never the ordinal index of independently aligned karaoke lines.
export function translationForLine(translation: LyricTranslation, source: string): string | undefined {
  const normalize = (text: string) => text.normalize("NFC").replace(/\s+/g, " ").trim();
  const matches = translation.source_lines.flatMap((text, id) => normalize(text) === normalize(source) ? [id] : []);
  const texts = matches.map(id => translation.lines.find(line => line.id === id)?.text).filter((text): text is string => text !== undefined);
  return texts.length && texts.every(text => text === texts[0]) ? texts[0] : undefined;
}

export default function LyricTranslationLine({ translation, source }: { translation: LyricTranslation; source: string }) {
  const text = translationForLine(translation, source);
  return text ? <span key={`${translation.target_language}:${source}`} className={styles.translation} lang={translation.target_language} dir="auto">{text}</span> : null;
}
