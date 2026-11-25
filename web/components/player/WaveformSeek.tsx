"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { usePlayer } from "./PlayerProvider";
import styles from "./WaveformSeek.module.css";
import { Button } from "../ui/button";

const stamp = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

/** When `showMelody` is supplied the caller owns the melody toggle and no inline button renders. */
export default function WaveformSeek({ compact = false, showMelody: controlledMelody }: { compact?: boolean; showMelody?: boolean }) {
  const { track, waveform, melody, duration, currentTime, seek } = usePlayer();
  const [localMelody, setShowMelody] = useState(true);
  const showMelody = controlledMelody ?? localMelody;
  const seekRef = useRef<HTMLSpanElement>(null);
  const [barCapacity, setBarCapacity] = useState(1);
  useEffect(() => {
    const element = seekRef.current;
    if (!element) return;
    // About 2.6px of ink and 1.4px of space per bar at any player size.
    const observer = new ResizeObserver(entries => {
      const width = entries[0]?.contentRect.width ?? 0;
      setBarCapacity(Math.max(1, Math.floor(width / 4)));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const maximum = Number.isFinite(duration) && duration > 0 ? duration : 0;
  const position = Math.max(0, Math.min(Number.isFinite(currentTime) ? currentTime : 0, maximum));
  const progress = maximum ? position / maximum * 100 : 0;
  const path = useMemo(() => {
    if (!waveform?.length) return "";
    const count = Math.min(barCapacity, waveform.length);
    return Array.from({ length: count }, (_, index) => {
      const start = Math.floor(index * waveform.length / count);
      const end = Math.floor((index + 1) * waveform.length / count);
      const bucket = waveform.slice(start, end);
      const energy = bucket.reduce((sum, value) => sum + value, 0) / bucket.length;
      const height = Math.max(1, energy * 44);
      const x = index * 1000 / count;
      return `M${x},${(48 - height) / 2}h${1000 / count * .65}v${height}h-${1000 / count * .65}Z`;
    }).join(" ");
  }, [waveform, barCapacity]);
  const melodyPath = useMemo(() => {
    if (!melody || !maximum || compact || !showMelody) return "";
    const pitches = melody.points.flatMap(point => point.pitch === null ? [] : [point.pitch]);
    if (!pitches.length) return "";
    const low = Math.min(...pitches), high = Math.max(...pitches);
    const span = Math.max(12, high - low);
    const center = (high + low) / 2;
    let connected = false;
    return melody.points.map(point => {
      if (point.pitch === null || point.time_seconds > maximum) { connected = false; return ""; }
      const command = connected ? "L" : "M";
      connected = true;
      return `${command}${point.time_seconds / maximum * 1000},${24 - (point.pitch - center) / span * 36}`;
    }).join(" ");
  }, [melody, maximum, compact, showMelody]);
  return <span className={styles.frame}><span ref={seekRef} className={`${styles.seek} ${compact ? styles.compact : ""}`} style={{ "--played": `${progress}%` } as CSSProperties}>
    {path ? <><svg viewBox="0 0 1000 48" preserveAspectRatio="none" aria-hidden="true"><path d={path} /></svg><svg className={styles.played} viewBox="0 0 1000 48" preserveAspectRatio="none" aria-hidden="true"><path d={path} /></svg></> : <span className={styles.fallback} aria-hidden="true" />}
    {melodyPath && <svg className={styles.melody} viewBox="0 0 1000 48" preserveAspectRatio="none" aria-hidden="true"><path d={melodyPath} vectorEffect="non-scaling-stroke" /></svg>}
    <input type="range" min={0} max={maximum} step={0.1} value={position} disabled={!track || !maximum} onChange={event => seek(Number(event.target.value))} aria-label="Seek" aria-valuetext={`${stamp(position)} of ${stamp(maximum)}`} />
  </span>{!compact && melody && controlledMelody === undefined && <Button variant="ghost" size="sm" type="button" className={styles.melodyToggle} aria-pressed={showMelody} onClick={() => setShowMelody(value => !value)} title="Pitch from precomputed hum-search analysis, not loudness">Melody · {melody.source.replaceAll("-", " ")} · {showMelody ? "on" : "off"}</Button>}</span>;
}
