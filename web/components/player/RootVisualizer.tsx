"use client";

import type { VisualFrame } from "./visualFeatures";
import * as THREE from "three";
import { useEffect, useRef } from "react";
import { PlaybackPreferences, readPlaybackPreferences } from "./playbackPreferences";
import { SongJourney } from "./songJourney";
import styles from "./RootVisualizer.module.css";

type Color = [number, number, number];
type Segment = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  bucket: number;
  life: number;
  generation: number;
  distance: number;
};
type Tip = {
  x: number;
  y: number;
  angle: number;
  bucket: number;
  energy: number;
  dormant: boolean;
  generation: number;
  distance: number;
};
type PaletteEvent = { active: boolean; palette: { waves: [Color, Color, Color] } | null };
type Reactivity = {
  bass: number;
  mid: number;
  treble: number;
  onset: boolean;
  bassAttack: number;
  midAttack: number;
  trebleAttack: number;
  timestamp?: number;
};

const BUCKETS = 16;
const MAX_SEGMENTS = 2400;
const GROWTH_STEP = 1 / 60;
const fallback: [Color, Color, Color] = [
  [0.23, 0.78, 0.72],
  [0.48, 0.35, 0.72],
  [0.84, 0.43, 0.5],
];
const mix = (left: Color, right: Color, amount: number): Color =>
  left.map((value, index) => value + (right[index] - value) * amount) as Color;

export default function RootVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: false,
        powerPreference: "high-performance",
      });
    } catch {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
    const rootGeometry = new THREE.BufferGeometry();
    // Six vertices form a narrow ribbon per segment. Distance is continuous
    // across segments and forks, so light stays attached to the root network.
    const rootPositions = new Float32Array(MAX_SEGMENTS * 18);
    const rootColors = new Float32Array(MAX_SEGMENTS * 18);
    const rootPaths = new Float32Array(MAX_SEGMENTS * 12);
    rootGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(rootPositions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    rootGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(rootColors, 3).setUsage(THREE.DynamicDrawUsage),
    );
    rootGeometry.setAttribute(
      "path",
      new THREE.BufferAttribute(rootPaths, 2).setUsage(THREE.DynamicDrawUsage),
    );
    rootGeometry.setDrawRange(0, 0);
    const rootMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
      side: THREE.DoubleSide,
      uniforms: { travel: { value: 0 }, beat: { value: 0 } },
      vertexShader: `
        attribute vec2 path;
        varying vec2 vPath;
        varying vec3 vColor;
        void main() {
          vPath = path;
          vColor = color;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float travel;
        uniform float beat;
        varying vec2 vPath;
        varying vec3 vColor;
        void main() {
          float edge = abs(vPath.y);
          float core = 1.0 - smoothstep(0.08, 0.38, edge);
          float halo = exp(-edge * edge * 5.0) * (1.0 - smoothstep(0.7, 1.0, edge));
          float phase = fract((vPath.x - travel) / 240.0);
          float head = min(phase, 1.0 - phase);
          float pulse = exp(-head * head * 420.0);
          float alpha = core * 0.4 + halo * 0.12
            + pulse * (core * 0.65 + halo * 0.36) * (0.65 + beat * 0.7);
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
    });
    const roots = new THREE.Mesh(rootGeometry, rootMaterial);
    roots.frustumCulled = false;
    scene.add(roots);

    let width = 1,
      height = 1,
      frame = 0,
      generation = 0,
      lastFrame = 0;
    let preferences = readPlaybackPreferences();
    let enabled = preferences.wavesEnabled && preferences.backdropPreset === "roots";
    let artwork: Color[] = fallback;
    let palette: [Color, Color, Color] = fallback;
    const journey = new SongJourney();
    let lastScene = 0,
      lastImpact = 0;
    let levels = new Float32Array(BUCKETS);
    const previous = new Float32Array(BUCKETS);
    const segments: Segment[] = [];
    const tips: Tip[] = [];
    const reactivity: Reactivity = {
      bass: 0,
      mid: 0,
      treble: 0,
      onset: false,
      bassAttack: 0,
      midAttack: 0,
      trebleAttack: 0,
    };
    let bpm = 0,
      beatFlash = 0,
      travel = 0,
      audioActive = false,
      drive = 0,
      growthClock = 0;

    const seed = (bucket: number, parent?: Segment) => {
      const edgeBias = Math.random();
      const x = parent
        ? parent.x2
        : edgeBias < 0.55
          ? Math.random() * width
          : Math.random() < 0.5
            ? 0
            : width;
      const y = parent ? parent.y2 : Math.random() * height;
      const angle = parent
        ? Math.atan2(parent.y2 - parent.y1, parent.x2 - parent.x1) + (Math.random() - 0.5) * 1.7
        : Math.random() * Math.PI * 2;
      tips.push({
        x,
        y,
        angle,
        bucket,
        energy: 0.2,
        dormant: false,
        generation,
        distance: parent
          ? parent.distance + Math.hypot(parent.x2 - parent.x1, parent.y2 - parent.y1)
          : Math.random() * 240,
      });
    };

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      width = Math.max(1, bounds.width);
      height = Math.max(1, bounds.height);
      renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
      renderer.setSize(width, height, false);
      camera.left = 0;
      camera.right = width;
      camera.top = height;
      camera.bottom = 0;
      camera.updateProjectionMatrix();
      if (!tips.length) for (let bucket = 0; bucket < BUCKETS; bucket += 1) seed(bucket);
    };

    const receiveSpectrum = (event: Event) => {
      const bins = (event as CustomEvent<VisualFrame>).detail.bands;
      const next = new Float32Array(BUCKETS);
      for (let bucket = 0; bucket < BUCKETS; bucket += 1) {
        const from = Math.max(1, Math.floor(Math.pow(bucket / BUCKETS, 2.15) * bins.length));
        const to = Math.max(
          from + 1,
          Math.floor(Math.pow((bucket + 1) / BUCKETS, 2.15) * bins.length),
        );
        let sum = 0;
        for (let index = from; index < Math.min(to, bins.length); index += 1) sum += bins[index];
        const response =
          bucket < 6
            ? preferences.bassReactivity
            : bucket < 12
              ? preferences.vocalReactivity
              : preferences.trebleReactivity;
        next[bucket] = Math.min(
          1,
          (sum / Math.max(1, Math.min(to, bins.length) - from)) * response,
        );
      }
      levels = next;
    };
    const receivePalette = (event: Event) => {
      const detail = (event as CustomEvent<PaletteEvent>).detail;
      if (detail.palette) artwork = detail.palette.waves;
    };
    const receiveReactivity = (event: Event) => {
      const detail = (event as CustomEvent<VisualFrame>).detail;
      journey.receive(detail, preferences);
      Object.assign(reactivity, detail);
      audioActive = detail.active;
      bpm = detail.bpm ?? 0;
      if (!detail.active) {
        levels.fill(0);
        previous.fill(0);
        beatFlash = 0;
        rootMaterial.uniforms.beat.value = 0;
      } else if (detail.beat) beatFlash = 1;
    };
    const changeTrack = () => {
      generation += 1;
      journey.reset();
      lastScene = 0;
    };
    const receivePreferences = (event: Event) => {
      preferences = (event as CustomEvent<PlaybackPreferences>).detail;
      enabled = preferences.wavesEnabled && preferences.backdropPreset === "roots";
      canvas.style.opacity = enabled ? "1" : "0";
    };
    canvas.style.opacity = enabled ? "1" : "0";

    const updateRootBuffers = () => {
      let vertex = 0;
      for (const segment of segments) {
        if (vertex >= MAX_SEGMENTS * 6) break;
        if (segment.life <= 0) continue;
        const position = segment.bucket / (BUCKETS - 1);
        const base =
          position < 0.5
            ? mix(palette[0], palette[1], position * 2)
            : mix(palette[1], palette[2], (position - 0.5) * 2);
        const brightness =
          (0.3 + levels[segment.bucket] * 0.7) * segment.life * (0.7 + drive * 0.6);
        const dx = segment.x2 - segment.x1,
          dy = segment.y2 - segment.y1;
        const length = Math.max(0.001, Math.hypot(dx, dy));
        const width = 2.6 + drive * 1.8;
        const nx = (-dy / length) * width,
          ny = (dx / length) * width;
        for (const corner of [0, 1, 2, 2, 1, 3]) {
          const end = corner >= 2;
          const side = corner % 2 ? 1 : -1;
          const offset = vertex * 3;
          rootPositions[offset] = (end ? segment.x2 : segment.x1) + nx * side;
          rootPositions[offset + 1] = (end ? segment.y2 : segment.y1) + ny * side;
          rootPositions[offset + 2] = 0;
          rootColors[offset] = base[0] * brightness;
          rootColors[offset + 1] = base[1] * brightness;
          rootColors[offset + 2] = base[2] * brightness;
          rootPaths[vertex * 2] = segment.distance + (end ? length : 0);
          rootPaths[vertex * 2 + 1] = side;
          vertex++;
        }
      }
      rootGeometry.setDrawRange(0, vertex);
      for (const attribute of Object.values(rootGeometry.attributes)) attribute.needsUpdate = true;
    };

    const draw = (now: number) => {
      frame = requestAnimationFrame(draw);
      const frameInterval =
        preferences.waveFrameRate === "uncapped" ? 0 : 1000 / Number(preferences.waveFrameRate);
      if (!enabled || now - lastFrame < frameInterval) return;
      const deltaSeconds = lastFrame
        ? Math.min(0.05, Math.max(0.001, (now - lastFrame) / 1000))
        : 1 / 60;
      lastFrame = now;
      const animationScale =
        preferences.animationSpeed === "slow"
          ? 0.65
          : preferences.animationSpeed === "fast"
            ? 1.45
            : 1;
      journey.step(deltaSeconds, animationScale, now / 1000);
      drive = journey.drive;
      palette = [journey.color(artwork, 0), journey.color(artwork, 1), journey.color(artwork, 2)];
      // A new scene grows a new network in its colours while the previous one fades.
      if (journey.scene !== lastScene) {
        lastScene = journey.scene;
        generation += 1;
      }
      // A bigger section landing sprouts fresh branches across the whole network.
      if (journey.impact > lastImpact + 0.2) {
        for (let bucket = 0; bucket < BUCKETS; bucket += 1) {
          const candidates = segments.filter(
            (segment) => segment.bucket === bucket && segment.life > 0.25,
          );
          for (let sprout = 0; sprout < 2; sprout += 1)
            seed(bucket, candidates[Math.floor(Math.random() * candidates.length)]);
        }
      }
      lastImpact = journey.impact;
      for (const segment of segments)
        segment.life -= deltaSeconds * (segment.generation < generation ? 0.36 : 0.03);
      beatFlash = Math.max(0, beatFlash - deltaSeconds * 2.8);
      while (segments.length && (segments[0].life <= 0 || segments.length > MAX_SEGMENTS))
        segments.shift();

      // Growth runs on a fixed 60 Hz clock, so speed, branching and network length do
      // not depend on the frame-rate cap. Long stalls drop steps rather than burst.
      growthClock = Math.min(growthClock + deltaSeconds, GROWTH_STEP * 4);
      for (; growthClock >= GROWTH_STEP; growthClock -= GROWTH_STEP) grow(animationScale);
      // Trim before uploading to the fixed-size GPU arrays.
      if (segments.length > MAX_SEGMENTS) segments.splice(0, segments.length - MAX_SEGMENTS);
      updateRootBuffers();
      // Integrating speed avoids phase jumps when the tempo estimate changes.
      if (audioActive) travel += deltaSeconds * (100 + bpm * 1.5) * animationScale;
      rootMaterial.uniforms.travel.value = travel;
      rootMaterial.uniforms.beat.value = beatFlash;
      renderer.render(scene, camera);
    };

    const grow = (animationScale: number) => {
      for (let bucket = 0; bucket < BUCKETS; bucket += 1) {
        const amplitude = levels[bucket],
          delta = amplitude - previous[bucket],
          attack = Math.max(0, delta);
        const activeTips = tips.filter(
          (tip) => tip.bucket === bucket && tip.generation === generation && !tip.dormant,
        );
        if ((amplitude > 0.1 || attack > 0.05) && !activeTips.length) {
          const candidates = segments.filter(
            (segment) => segment.bucket === bucket && segment.life > 0.25,
          );
          seed(bucket, candidates[Math.floor(Math.random() * candidates.length)]);
        }
        for (const tip of activeTips) {
          if (amplitude < 0.05) {
            tip.energy *= 0.9;
            if (tip.energy < 0.04) tip.dormant = true;
            continue;
          }
          tip.energy += (amplitude - tip.energy) * 0.35;
          tip.angle += delta * 6 + (Math.random() - 0.5) * (0.06 + amplitude * 0.22 + attack * 1.2);
          const tempoDrive = 0.55 + Math.min(1.75, bpm / 105);
          const speed =
            (0.25 + Math.pow(amplitude, 1.4) * 11 + attack * 32 + reactivity.bassAttack * 10) *
            tempoDrive *
            animationScale *
            (0.45 + drive * 0.9 + journey.impact * 0.6);
          let x = tip.x + Math.cos(tip.angle) * speed,
            y = tip.y + Math.sin(tip.angle) * speed;
          segments.push({
            x1: tip.x,
            y1: tip.y,
            x2: x,
            y2: y,
            bucket,
            life: 1,
            generation,
            distance: tip.distance,
          });
          tip.distance += Math.hypot(x - tip.x, y - tip.y);
          if (x < 0) x += width;
          else if (x > width) x -= width;
          if (y < 0) y += height;
          else if (y > height) y -= height;
          tip.x = x;
          tip.y = y;
          if (Math.random() < amplitude * 0.03 + attack * 0.5)
            seed(bucket, segments[segments.length - 1]);
        }
        previous[bucket] = amplitude;
      }
      if (tips.length > BUCKETS * 6) tips.splice(0, tips.length - BUCKETS * 6);
      // Growth can cross capacity within a step; keep the list bounded as it grows.
      if (segments.length > MAX_SEGMENTS) segments.splice(0, segments.length - MAX_SEGMENTS);
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("echora:visual-frame", receiveSpectrum);
    window.addEventListener("echora:track-palette", receivePalette);
    window.addEventListener("echora:visual-frame", receiveReactivity);
    window.addEventListener("echora:track-change", changeTrack);
    window.addEventListener("echora:playback-preferences", receivePreferences);
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      window.removeEventListener("echora:visual-frame", receiveSpectrum);
      window.removeEventListener("echora:track-palette", receivePalette);
      window.removeEventListener("echora:visual-frame", receiveReactivity);
      window.removeEventListener("echora:track-change", changeTrack);
      window.removeEventListener("echora:playback-preferences", receivePreferences);
      rootGeometry.dispose();
      rootMaterial.dispose();
      renderer.dispose();
    };
  }, []);

  return <canvas ref={canvasRef} className={styles.roots} aria-hidden="true" />;
}
