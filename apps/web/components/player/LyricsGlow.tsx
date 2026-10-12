"use client";

import { useEffect, useRef, type RefObject } from "react";
import styles from "./LyricsGlow.module.css";

type Spot = { x: number; y: number; w: number; h: number };
const spot = (): Spot => ({ x: 0, y: 0, w: 1, h: 1 });

/**
 * Text remains in the DOM. This layer only paints light behind its bounds.
 *
 * Each light is one element with a soft radial gradient, moved with `transform` and faded with
 * `opacity`, which the compositor animates without repainting. (A full-screen WebGL canvas
 * redrawn every frame cost most of the lyrics screen's CPU and GPU time.)
 */
export default function LyricsGlow({
  container,
  playing,
}: {
  container: RefObject<HTMLElement | null>;
  playing: boolean;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  const playingRef = useRef(playing);
  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);
  useEffect(() => {
    const layer = layerRef.current;
    const root = container.current;
    if (!layer || !root) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (reduced.matches) return;
    // Two groups of eight light spots: the singing line, and the line it just left,
    // which fades out in place while the new one fades in.
    const SPOTS = 8;
    const elements = Array.from({ length: SPOTS * 2 }, () => {
      const element = document.createElement("div");
      element.className = styles.spot;
      layer.appendChild(element);
      return element;
    });
    // Spots live relative to their lyric line, so scrolling carries them with the text
    // exactly; only movement within a line (word to word) glides.
    type Group = {
      line: HTMLElement | null;
      rel: Spot[];
      target: Spot[];
      count: number;
      weight: number;
    };
    const group = (): Group => ({
      line: null,
      count: 0,
      weight: 0,
      rel: Array.from({ length: SPOTS }, spot),
      target: Array.from({ length: SPOTS }, spot),
    });
    let current = group(),
      previous = group();
    let frame = 0,
      last = 0,
      lastTint = 0,
      clock = 0,
      visible = false,
      cleared = true;
    // Text boxes span the font's full ascent and descent, whose middle sits above the letters'
    // visual middle. Measure the real glyphs once per font and text, and move the glow down by the gap.
    const metrics = document.createElement("canvas").getContext("2d");
    const offsets = new Map<string, number>();
    // Per element, so the style isn't recomputed every frame while the same line glows.
    const elementOffsets = new WeakMap<HTMLElement, { text: string; value: number }>();
    const visualOffset = (element: HTMLElement) => {
      if (!metrics) return 0;
      const text = (element.textContent || "").trim().slice(0, 80);
      const seen = elementOffsets.get(element);
      if (seen && seen.text === text) return seen.value;
      const style = getComputedStyle(element);
      const key = `${style.font}|${text}`;
      let value = offsets.get(key);
      if (value === undefined) {
        metrics.font = style.font;
        const glyphs = metrics.measureText(text || "Hx");
        const offset =
          (glyphs.fontBoundingBoxAscent -
            glyphs.fontBoundingBoxDescent -
            (glyphs.actualBoundingBoxAscent - glyphs.actualBoundingBoxDescent)) /
          2;
        value = Number.isFinite(offset) ? Math.max(-12, Math.min(12, offset)) : 0;
        if (offsets.size > 200) offsets.clear();
        offsets.set(key, value);
      }
      elementOffsets.set(element, { text, value });
      return value;
    };
    const measure = () => {
      const active = root.querySelectorAll<HTMLElement>('[data-lyric-singing="true"]');
      let boxes: DOMRect[] = [];
      for (const element of active) boxes.push(...Array.from(element.getClientRects()));
      boxes = boxes.filter((box) => box.width > 0 && box.height > 0).slice(0, SPOTS);
      // Only timed karaoke fragments emit light. Rests intentionally have no glow.
      visible = boxes.length > 0;
      if (!visible) return;
      const line = active[0]?.closest<HTMLElement>("[data-index]") ?? null;
      if (!line) {
        visible = false;
        return;
      }
      const anchor = line.getBoundingClientRect();
      const textElement = active[0]?.parentElement;
      const shift = textElement ? visualOffset(textElement) : 0;
      if (line !== current.line) {
        // A new line: the old glow keeps fading on its own line; the new one starts in place.
        // Returning to the line still fading out (a quick seek back) resumes its glow.
        const returning = line === previous.line;
        [previous, current] = [current, previous];
        if (!returning) {
          current.line = line;
          current.weight = 0;
        }
      }
      current.count = boxes.length;
      boxes.forEach((box, index) => {
        const target = current.target[index];
        target.x = box.x - anchor.x + box.width / 2;
        target.y = box.y - anchor.y + box.height / 2 + shift;
        target.w = Math.min(180, box.width / 2 + 24);
        target.h = box.height / 2 + 22;
      });
      if (current.weight < 0.02)
        current.target.forEach((target, index) => Object.assign(current.rel[index], target));
    };
    const place = (from: Group, offset: number, origin: DOMRect) => {
      const anchor = from.line?.isConnected ? from.line.getBoundingClientRect() : null;
      for (let index = 0; index < SPOTS; index++) {
        const element = elements[offset + index];
        const lit = anchor && index < from.count ? from.weight : 0;
        if (!lit || !anchor) {
          if (element.style.opacity !== "0") element.style.opacity = "0";
          continue;
        }
        const rel = from.rel[index];
        const i = offset + index;
        const driftX = Math.sin(clock * 0.8 + i) * 1.5;
        const driftY = Math.cos(clock * 0.65 + i) * 1.5;
        const x = anchor.x - origin.x + rel.x + driftX;
        const y = anchor.y - origin.y + rel.y + driftY;
        // A 100 px gradient square scaled to the spot's half-extents (the old shader's falloff).
        element.style.transform = `translate(${(x - rel.w).toFixed(1)}px, ${(y - rel.h).toFixed(1)}px) scale(${((rel.w * 2) / 100).toFixed(3)}, ${((rel.h * 2) / 100).toFixed(3)})`;
        element.style.opacity = lit.toFixed(3);
      }
    };
    const clear = () => {
      if (cleared) return;
      elements.forEach((element) => (element.style.opacity = "0"));
      cleared = true;
    };
    const draw = (now: number) => {
      frame = requestAnimationFrame(draw);
      if (document.hidden || reduced.matches) {
        last = now;
        return;
      }
      // 30 fps, in step with the karaoke wipe it follows.
      if (now - last < 1000 / 30 - 4) return;
      const dt = Math.min(0.08, (now - last) / 1000 || 0.033);
      last = now;
      // Nothing to light and nothing still fading: stop measuring until the song plays again
      // (pauses and instrumental breaks cost almost nothing).
      if (!playingRef.current && current.weight < 0.005 && previous.weight < 0.005) return clear();
      measure();
      if (!visible && current.weight < 0.005 && previous.weight < 0.005) return clear();
      cleared = false;
      if (now - lastTint >= 500) {
        lastTint = now;
        const color = getComputedStyle(root).getPropertyValue("--accent-rgb").trim();
        if (color.split(/\s+/).length === 3) layer.style.setProperty("--glow-rgb", color);
      }
      const playing = playingRef.current;
      current.weight += ((visible && playing ? 1 : 0) - current.weight) * (1 - Math.exp(-dt * 10));
      previous.weight += (0 - previous.weight) * (1 - Math.exp(-dt * 12));
      const glide = 1 - Math.exp(-dt * 16);
      current.rel.forEach((rel, index) => {
        const target = current.target[index];
        rel.x += (target.x - rel.x) * glide;
        rel.y += (target.y - rel.y) * glide;
        rel.w += (target.w - rel.w) * glide;
        rel.h += (target.h - rel.h) * glide;
      });
      clock += dt;
      const origin = layer.getBoundingClientRect();
      place(current, 0, origin);
      place(previous, SPOTS, origin);
    };
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      elements.forEach((element) => element.remove());
    };
  }, [container]);
  return <div ref={layerRef} className={styles.glow} aria-hidden="true" />;
}
