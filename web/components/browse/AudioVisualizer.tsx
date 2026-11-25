"use client";

import { useEffect, useRef } from "react";
import styles from "./AudioVisualizer.module.css";

export default function AudioVisualizer({ stream }: { stream: MediaStream }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.72;
    const source = context.createMediaStreamSource(stream);
    const frequencies = new Uint8Array(analyser.frequencyBinCount);
    source.connect(analyser);
    let frame = 0;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(bounds.width * scale));
      const height = Math.max(1, Math.round(bounds.height * scale));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      analyser.getByteFrequencyData(frequencies);
      const drawing = canvas.getContext("2d");
      if (!drawing) return;
      drawing.clearRect(0, 0, width, height);
      drawing.fillStyle = "rgba(255, 137, 146, 0.5)";
      const bars = 22;
      const gap = Math.max(1, Math.round(2 * scale));
      const barWidth = (width - gap * (bars - 1)) / bars;
      for (let index = 0; index < bars; index += 1) {
        const bin = 2 + Math.round(index / (bars - 1) * 46);
        const strength = Math.max(0.08, frequencies[bin] / 255);
        const barHeight = Math.max(2 * scale, strength * height * 0.9);
        drawing.fillRect(index * (barWidth + gap), height - barHeight, barWidth, barHeight);
      }
      frame = window.requestAnimationFrame(draw);
    };

    draw();
    return () => {
      window.cancelAnimationFrame(frame);
      source.disconnect();
      analyser.disconnect();
      void context.close();
    };
  }, [stream]);

  return <canvas ref={canvasRef} className={styles.visualizer} aria-hidden="true" />;
}
