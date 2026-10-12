"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { fragmentProgress, lyricWordIsRtl, type LyricFragment } from "./lyricWords";
import styles from "./KaraokeLine.module.css";

type Box = { x: number; y: number; width: number; height: number };
type Geometry = { width: number; height: number; fragments: Box[][]; source: LyricFragment[] };

/** A playback clock: `now()` in milliseconds, `subscribe` called whenever it moves. */
export type KaraokeClock = { subscribe: (listener: () => void) => () => void; now: () => number };

type Props = {
  fragments: LyricFragment[];
  /** Time to paint without a clock (static rendering, tests). */
  now?: number;
  /** Follows the clock every frame by writing the clip path directly, not through React. */
  clock?: KaraokeClock;
  active: boolean;
  highlightStyle: "lava" | "syllable";
};

type Paint = { path: string; glows: Box[] };

/** The sung part of each fragment as one clip path, and where the glow should sit. */
function paintAt(
  geometry: Geometry | null,
  fragments: LyricFragment[],
  now: number,
  active: boolean,
  highlightStyle: "lava" | "syllable",
): Paint {
  if (!geometry || geometry.source !== fragments) return { path: "", glows: [] };
  const glows: Box[] = [];
  const path = geometry.fragments
    .flatMap((boxes, index) => {
      const fragment = fragments[index];
      const { syllable } = fragment;
      const fragmentRtl = lyricWordIsRtl(fragment.text);
      const singing = active && now >= syllable.start_ms && now < syllable.end_ms;
      const progress =
        now >= syllable.end_ms
          ? 100
          : singing
            ? highlightStyle === "syllable"
              ? 100
              : fragmentProgress(now, fragment)
            : 0;
      const total = boxes.reduce((sum, box) => sum + box.width, 0);
      if (singing) {
        if (highlightStyle === "syllable") glows.push(...boxes);
        else {
          let left = (total * fragmentProgress(now, fragment)) / 100;
          for (const box of boxes) {
            if (left <= box.width) {
              const edge = fragmentRtl ? box.x + box.width - left : box.x + left;
              glows.push({ x: edge, y: box.y + box.height / 2, width: 1, height: 1 });
              break;
            }
            left -= box.width;
          }
        }
      }
      let remaining = (total * progress) / 100;
      return boxes.map((box) => {
        const fill = Math.max(0, Math.min(box.width, remaining));
        remaining -= box.width;
        if (fill <= 0) return "";
        const top = box.y - 2,
          bottom = box.y + box.height + 2;
        const side = fragmentRtl ? box.x + box.width : box.x;
        const edge = side + (fragmentRtl ? -fill : fill);
        const amplitude = Math.min(5, fill * 0.15, (box.width - fill) * 0.15);
        if (amplitude <= 0)
          return `M${side},${top} L${edge.toFixed(2)},${top} L${edge.toFixed(2)},${bottom} L${side},${bottom} Z`;
        const boundary = Array.from({ length: 13 }, (_, point) => {
          const y = top + ((bottom - top) * point) / 12;
          const x =
            edge + Math.sin((point / 12) * Math.PI * 2 + now / 320 + index * 0.8) * amplitude;
          return `L${x.toFixed(2)},${y.toFixed(2)}`;
        }).join(" ");
        return `M${side},${top} ${boundary} L${side},${bottom} Z`;
      });
    })
    .filter(Boolean)
    .join(" ");
  return { path, glows };
}

/** Native text layout owns shaping, wrapping, punctuation, and bidirectional runs. */
export default function KaraokeLine({ fragments, now = 0, clock, active, highlightStyle }: Props) {
  const root = useRef<HTMLSpanElement>(null);
  const base = useRef<HTMLSpanElement>(null);
  const paintLayer = useRef<HTMLSpanElement>(null);
  const clipPath = useRef<SVGPathElement>(null);
  const glowLayer = useRef<HTMLSpanElement>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const clipId = `karaoke-clip-${useId().replace(/[^\w-]/g, "")}`;
  const text = fragments.map((fragment) => fragment.text).join("");
  useLayoutEffect(() => {
    const element = root.current;
    const node = base.current?.firstChild;
    if (!element || !node) return;
    let disposed = false;
    const measure = () => {
      if (disposed) return;
      const bounds = element.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      // Ignore entrance-animation transforms when converting viewport bounds to local pixels.
      const scaleX = bounds.width / element.offsetWidth;
      const scaleY = bounds.height / element.offsetHeight;
      let offset = 0;
      const boxes = fragments.map((fragment) => {
        const range = document.createRange();
        range.setStart(node, offset);
        offset += fragment.text.length;
        range.setEnd(node, offset);
        if (!fragment.text.trim()) return [];
        return Array.from(range.getClientRects()).map((box) => ({
          x: (box.left - bounds.left) / scaleX,
          y: (box.top - bounds.top) / scaleY,
          width: box.width / scaleX,
          height: box.height / scaleY,
        }));
      });
      setGeometry({
        width: element.offsetWidth,
        height: element.offsetHeight,
        fragments: boxes,
        source: fragments,
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    void document.fonts?.ready.then(measure);
    return () => {
      disposed = true;
      observer.disconnect();
    };
  }, [fragments]);

  // Live lines: one React render per line or layout change; every frame only touches the clip
  // path and the glow markers, and only when they actually change.
  useEffect(() => {
    if (!clock) return;
    let lastPath: string | null = null;
    let lastGlows = "";
    const update = () => {
      const { path, glows } = paintAt(geometry, fragments, clock.now(), active, highlightStyle);
      if (path !== lastPath) {
        lastPath = path;
        clipPath.current?.setAttribute("d", path);
        const layer = paintLayer.current;
        if (layer) {
          layer.style.clipPath = path ? `url(#${clipId})` : "";
          layer.style.visibility = path ? "visible" : "hidden";
        }
      }
      const glowKey = glows.map((box) => `${box.x},${box.y},${box.width},${box.height}`).join(";");
      const container = glowLayer.current;
      if (glowKey === lastGlows || !container) return;
      lastGlows = glowKey;
      // Reuse marker elements; LyricsGlow finds the visible ones by their attribute.
      while (container.children.length < glows.length) {
        const marker = document.createElement("span");
        marker.className = styles.glowTarget;
        marker.setAttribute("aria-hidden", "true");
        container.appendChild(marker);
      }
      Array.from(container.children).forEach((child, index) => {
        const marker = child as HTMLElement;
        const box = glows[index];
        if (!box) {
          marker.removeAttribute("data-lyric-singing");
          marker.style.display = "none";
          return;
        }
        marker.setAttribute("data-lyric-singing", "true");
        marker.style.display = "";
        marker.style.left = `${box.x}px`;
        marker.style.top = `${box.y}px`;
        marker.style.width = `${box.width}px`;
        marker.style.height = `${box.height}px`;
      });
    };
    update();
    return clock.subscribe(update);
  }, [clock, geometry, fragments, active, highlightStyle, clipId]);

  // Without a clock (static rendering), paint once from `now`.
  const initial = clock
    ? { path: "", glows: [] as Box[] }
    : paintAt(geometry, fragments, now, active, highlightStyle);
  const paint: CSSProperties = {
    clipPath: initial.path ? `url(#${clipId})` : undefined,
    visibility: initial.path ? "visible" : "hidden",
  };

  return (
    <span ref={root} dir="auto" className={styles.line}>
      <span ref={base}>{text}</span>
      <span ref={paintLayer} className={styles.paint} aria-hidden="true" style={paint}>
        {text}
      </span>
      <svg className={styles.clipDefs} aria-hidden="true" focusable="false">
        <clipPath id={clipId} clipPathUnits="userSpaceOnUse">
          <path ref={clipPath} d={initial.path} />
        </clipPath>
      </svg>
      <span ref={glowLayer} aria-hidden="true">
        {initial.glows.map((box, index) => (
          <span
            key={index}
            aria-hidden="true"
            data-lyric-singing="true"
            className={styles.glowTarget}
            style={{ left: box.x, top: box.y, width: box.width, height: box.height }}
          />
        ))}
      </span>
    </span>
  );
}
