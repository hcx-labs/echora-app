"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { Kawarp } from "@kawarp/core";
import styles from "./FullscreenPlayer.module.css";

/**
 * Stage view's background: the cover, blurred and slowly domain-warped by Kawarp (@kawarp/core,
 * MIT, Better Lyrics). The same renderer Radiant Lyrics uses for its cover backdrops.
 *
 * Cost is what kept an earlier full-screen WebGL layer (LyricsGlow) out: the canvas draws at a
 * quarter of its CSS size (the blur hides it), at most 30 frames a second, and stops drawing
 * once playback has paused and the motion has eased to rest. The blur itself runs once per cover.
 * Without WebGL, the CSS backdrop (two drifting blurred copies of the cover) is used instead.
 */
const FRAME_MS = 1000 / 30;
const RESOLUTION = 0.25;
const MAX_WIDTH = 640;
const TRANSITION_MS = 1200;

export function StageBackdrop({
  cover,
  playing,
  coverLoaded,
}: {
  cover: string;
  playing: boolean;
  /** The parent's own check that the cover has decoded; used only by the CSS fallback. */
  coverLoaded: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engine = useRef<Kawarp | null>(null);
  const wake = useRef<(forMs?: number) => void>(() => undefined);
  const playingRef = useRef(playing);
  const [failed, setFailed] = useState(false);
  const [loaded, setLoaded] = useState("");
  // Bumped once the renderer exists, so the cover effect runs again.
  const [ready, setReady] = useState(0);

  // Create the renderer once; it lives as long as Stage view is showing.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let disposed = false;
    let frame = 0;
    let last = 0;
    let time = Math.random() * 100;
    let speed = playingRef.current ? 1 : 0;
    // Draw until this moment even when paused: a cover crossfade or a resize needs frames.
    let busyUntil = 0;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)");
    let observer: ResizeObserver | undefined;

    const size = () => {
      const scale = Math.min(RESOLUTION, MAX_WIDTH / Math.max(1, canvas.clientWidth));
      canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale));
      canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale));
      engine.current?.resize();
    };
    const tick = (now: number) => {
      frame = 0;
      const kawarp = engine.current;
      if (disposed || !kawarp) return;
      if (now - last >= FRAME_MS) {
        const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
        last = now;
        // Ease to a stop on pause and back up on play, rather than freezing mid-motion.
        speed += ((playingRef.current && !still.matches ? 1 : 0) - speed) * Math.min(1, dt * 2.5);
        time += dt * speed;
        kawarp.renderFrame(time);
      }
      if (speed > 0.002 || playingRef.current || now < busyUntil) {
        frame = requestAnimationFrame(tick);
      } else {
        last = 0;
      }
    };
    wake.current = (forMs = 0) => {
      busyUntil = Math.max(busyUntil, performance.now() + forMs);
      if (!frame && !disposed) frame = requestAnimationFrame(tick);
    };

    import("@kawarp/core")
      .then(({ Kawarp }) => {
        if (disposed) return;
        engine.current = new Kawarp(canvas, {
          transitionDuration: TRANSITION_MS,
          animationSpeed: 0.6,
          warpIntensity: 0.9,
          saturation: 1.4,
          dithering: 0.01,
        });
        size();
        observer = new ResizeObserver(() => {
          size();
          wake.current(100);
        });
        observer.observe(canvas);
        setLoaded("");
        setReady((value) => value + 1);
      })
      .catch(() => {
        if (!disposed) setFailed(true);
      });
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer?.disconnect();
      engine.current?.dispose();
      engine.current = null;
    };
  }, []);

  useEffect(() => {
    const kawarp = engine.current;
    if (!kawarp || !cover) return;
    let current = true;
    kawarp
      .loadImage(cover)
      .then(() => {
        if (!current) return;
        setLoaded(cover);
        wake.current(TRANSITION_MS + 200);
      })
      .catch(() => undefined);
    return () => {
      current = false;
    };
  }, [cover, ready]);

  useEffect(() => {
    playingRef.current = playing;
    // Paused: keep drawing while the motion eases out.
    wake.current(playing ? 0 : 2500);
  }, [playing]);

  if (failed)
    return (
      <div
        className={styles.stageBackdrop}
        style={{ "--stage-cover": `url("${cover}")` } as CSSProperties}
        data-playing={playing}
        data-loaded={coverLoaded}
        aria-hidden="true"
      />
    );
  return (
    <div className={`${styles.stageBackdrop} ${styles.stageShader}`} aria-hidden="true">
      <canvas ref={canvasRef} data-loaded={Boolean(loaded)} />
    </div>
  );
}
