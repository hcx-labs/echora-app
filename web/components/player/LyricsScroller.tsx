"use client";

import { useEffect, useLayoutEffect, useRef, type KeyboardEvent, type ReactNode } from "react";
import { springSettled, springStep, type SpringState } from "./springScroll";

/** After the reader scrolls by hand, wait this long before following again. */
const MANUAL_PAUSE_MS = 3500;

/**
 * Keeps the active lyric line centred with a spring-driven scroll. The first
 * position is applied instantly; later line changes glide and can retarget
 * mid-motion. Wheel, touch or keyboard scrolling pauses following, which then
 * resumes on its own.
 */
/** `layoutKey` changes when line sizes change (text size, translations) so the list re-centres. */
export default function LyricsScroller({ activeIndex, layoutKey, className, children }: { activeIndex: number; layoutKey?: string; className?: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const spring = useRef<SpringState | null>(null);
  const frame = useRef(0);
  const manualUntil = useRef(0);
  const resumeTimer = useRef(0);
  const placed = useRef(false);
  const index = useRef(activeIndex);

  // Recompute where the active line should sit, then glide (or jump) there.
  const follow = useRef((instant: boolean) => {
    const container = ref.current;
    if (!container || Date.now() < manualUntil.current) return;
    const line = container.querySelector<HTMLElement>(`[data-index="${Math.max(0, index.current)}"]`);
    if (!line) return;
    const max = container.scrollHeight - container.clientHeight;
    const target = Math.max(0, Math.min(max, line.offsetTop - container.clientHeight / 2 + line.offsetHeight / 2));
    if (instant || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      cancelAnimationFrame(frame.current); frame.current = 0;
      spring.current = null;
      container.scrollTop = target;
      return;
    }
    // Start from wherever the list is now, keeping velocity if already moving.
    spring.current = { position: spring.current?.position ?? container.scrollTop, velocity: spring.current?.velocity ?? 0 };
    if (frame.current) return;
    let last = performance.now();
    const tick = (now: number) => {
      const state = spring.current;
      const current = ref.current;
      if (!state || !current) { frame.current = 0; return; }
      const goalLine = current.querySelector<HTMLElement>(`[data-index="${Math.max(0, index.current)}"]`);
      const goal = goalLine ? Math.max(0, Math.min(current.scrollHeight - current.clientHeight, goalLine.offsetTop - current.clientHeight / 2 + goalLine.offsetHeight / 2)) : target;
      const next = springStep(state, goal, Math.min(.05, (now - last) / 1000));
      last = now;
      if (springSettled(next, goal)) {
        current.scrollTop = goal; spring.current = null; frame.current = 0;
        return;
      }
      spring.current = next;
      current.scrollTop = next.position;
      frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
  });

  // First placement is instant so opening the view never scrolls from the top.
  useLayoutEffect(() => {
    index.current = activeIndex;
    follow.current(!placed.current);
    placed.current = true;
  }, [activeIndex]);

  // Line sizes changed: re-centre without animating.
  useLayoutEffect(() => { if (placed.current) follow.current(true); }, [layoutKey]);

  // The panel itself resized (window, layout): re-centre without animating.
  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    const observer = new ResizeObserver(() => follow.current(true));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => () => { cancelAnimationFrame(frame.current); window.clearTimeout(resumeTimer.current); }, []);

  function pause() {
    cancelAnimationFrame(frame.current); frame.current = 0; spring.current = null;
    manualUntil.current = Date.now() + MANUAL_PAUSE_MS;
    window.clearTimeout(resumeTimer.current);
    resumeTimer.current = window.setTimeout(() => follow.current(false), MANUAL_PAUSE_MS + 50);
  }
  function key(event: KeyboardEvent) {
    if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) pause();
  }

  return <div ref={ref} className={className} onWheel={pause} onTouchMove={pause} onKeyDown={key}>{children}</div>;
}
