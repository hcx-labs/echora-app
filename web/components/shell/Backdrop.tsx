/**
 * Backdrop presets inspired by the Void audio visualizer (https://github.com/epxweb/void-visualizer)
 * by epxstudio (taniguchi@epxstudio.com), licensed under the GNU General Public License.
 * Concepts adapted under GPL: the "Void tunnel", "Digital curtain", and "ASCII dance"
 * presets are reimplementations of their scenes (InfiniteTunnel, DigitalCurtain,
 * ASCIIDance) on Echora's 2D canvas pipeline, with Echora palette, tempo,
 * waveform-history, and curvature extensions.
 */

"use client";

import { useEffect, useRef, useState } from "react";
import {
  BackdropPreset,
  PlaybackPreferences,
  readPlaybackPreferences,
} from "../player/playbackPreferences";
import RootVisualizer from "../player/RootVisualizer";
import SignalVisualizer from "../player/SignalVisualizer";
import VoidTunnelVisualizer from "../player/VoidTunnelVisualizer";
import CloudVisualizer from "../player/CloudVisualizer";
import MeshGridVisualizer from "../player/MeshGridVisualizer";
import LightningFallVisualizer from "../player/LightningFallVisualizer";
import WaterDropVisualizer from "../player/WaterDropVisualizer";
import { SMALL_SCREEN_QUERY, applySmallScreenEffects } from "../player/smallScreen";
import styles from "./Backdrop.module.css";
import { advanceWaveSpring } from "./waveSpring";
import { SongJourney } from "../player/songJourney";
import type { VisualFrame } from "../player/visualFeatures";

type RenderLayer = { render: (time: number) => void };
type Color = [number, number, number];
type SplineSettings = {
  flowSpeed: number;
  bandAmplitude: number;
  waveHeightScale: number;
  brightness: number;
  opacity: number;
  layerAmplitudes: number[];
  layerColors: Color[];
  colorR: number;
  colorG: number;
  colorB: number;
};
type ParticleSettings = {
  count: number;
  flowSpeed: number;
  opacity: number;
  sizeBase: number;
  sizeVar: number;
};
type Reactivity = {
  bass: number;
  mid: number;
  treble: number;
  level: number;
  onset?: boolean;
  timestamp?: number;
  bassAttack?: number;
  midAttack?: number;
  trebleAttack?: number;
  drive?: number;
  impact?: number;
};
type XmbWindow = Window & {
  createSplineLayer?: (gl: WebGL2RenderingContext, canvas: HTMLCanvasElement) => RenderLayer;
  createParticlesLayer?: (gl: WebGL2RenderingContext, canvas: HTMLCanvasElement) => RenderLayer;
  SPLINE_SETTINGS?: SplineSettings;
  PARTICLE_SETTINGS?: ParticleSettings;
};

const scripts = [
  "spline-settings.js",
  "particles-settings.js",
  "spline-reverse.js",
  "spline.js",
  "particles.js",
];
const scriptLoads = new Map<string, Promise<void>>();

function loadScript(file: string) {
  const existing = scriptLoads.get(file);
  if (existing) return existing;
  const promise = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `/vendor/ps3-xmb/${file}`;
    script.async = false;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Could not load ${file}`));
    document.head.appendChild(script);
  });
  scriptLoads.set(file, promise);
  return promise;
}

const DEFAULT_BASE: Color = [37, 89, 179];
const DEFAULT_WAVES: [Color, Color, Color] = [
  [0.48, 0.98, 0.92],
  [0.76, 0.66, 1],
  [0.55, 0.8, 1],
];

function sceneBackground(
  ctx: CanvasRenderingContext2D,
  base: Color,
  width: number,
  height: number,
) {
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  const top = base.map((value, channel) => value * (channel === 2 ? 0.09 * 1.2 : 0.09));
  const bottom = base.map((value) => value * 0.62);
  gradient.addColorStop(0, `rgb(${top.map((value) => Math.round(value)).join(",")})`);
  gradient.addColorStop(1, `rgb(${bottom.map((value) => Math.round(value)).join(",")})`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
}

type CurtainCell = {
  digit: number;
  lastChange: number;
  changeSpeed: number;
  speedMult: number;
  flash: number;
  darken: number;
};
type CurtainState = { cells: CurtainCell[]; cols: number; rows: number };
const SEGMENT_MAP: Record<number, number[]> = {
  0: [1, 1, 1, 1, 1, 1, 0],
  1: [0, 1, 1, 0, 0, 0, 0],
  2: [1, 1, 0, 1, 1, 0, 1],
  3: [1, 1, 1, 1, 0, 0, 1],
  4: [0, 1, 1, 0, 0, 1, 1],
  5: [1, 0, 1, 1, 0, 1, 1],
  6: [1, 0, 1, 1, 1, 1, 1],
  7: [1, 1, 1, 0, 0, 0, 0],
  8: [1, 1, 1, 1, 1, 1, 1],
  9: [1, 1, 1, 1, 0, 1, 1],
};
const ASCII_CHARS = "♪♫★☆◆◇○●◐◑░▒▓█▪▫ABCDEФ0123456789!?#$%&<>*+=-~";
let asciiAtlas: HTMLCanvasElement | null = null;
let asciiRandoms: Float32Array | null = null;

function asciiAtlasCanvas(): HTMLCanvasElement {
  if (asciiAtlas) return asciiAtlas;
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let index = 0; index < 64; index++) {
    ctx.font = `${index % 2 ? 300 : 500} ${36 + (index % 3) * 4}px "DM Mono", monospace`;
    ctx.fillText(
      ASCII_CHARS[index % ASCII_CHARS.length],
      (index % 8) * 32 + 16,
      Math.floor(index / 8) * 32 + 17,
    );
  }
  asciiAtlas = canvas;
  return canvas;
}

// The glyph atlas is white; drawImage ignores fillStyle, so each colour gets a
// tinted copy. Copies refresh at most a few times a second as colours drift.
const tintedAtlases = new Map<number, { canvas: HTMLCanvasElement; color: Color; at: number }>();
function tintedAtlas(slot: number, color: Color, now: number): HTMLCanvasElement {
  const cached = tintedAtlases.get(slot);
  const drift = cached
    ? Math.max(...color.map((value, index) => Math.abs(value - cached.color[index])))
    : 1;
  if (cached && (drift < 0.02 || now - cached.at < 0.25)) return cached.canvas;
  const canvas = cached?.canvas ?? document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d")!;
  ctx.globalCompositeOperation = "copy";
  ctx.drawImage(asciiAtlasCanvas(), 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = `rgb(${color.map((value) => Math.round(Math.max(0, Math.min(1, value)) * 255)).join(",")})`;
  ctx.fillRect(0, 0, 256, 256);
  tintedAtlases.set(slot, { canvas, color: [...color] as Color, at: now });
  return canvas;
}

function curtainCellsFor(
  cols: number,
  rows: number,
  elapsed: number,
  existing: CurtainState | null,
): CurtainState {
  if (existing && existing.cols === cols && existing.rows === rows) return existing;
  const cells: CurtainCell[] = Array.from({ length: cols * rows }, () => ({
    digit: Math.floor(Math.random() * 10),
    lastChange: elapsed,
    changeSpeed: 0.1 + Math.random() * 0.5,
    speedMult: 1,
    flash: 0,
    darken: 1,
  }));
  return { cells, cols, rows };
}

function presetScene(
  ctx: CanvasRenderingContext2D,
  preset: BackdropPreset,
  width: number,
  height: number,
  elapsed: number,
  clock: number,
  delta: number,
  reactive: Reactivity,
  background: Color,
  waves: [Color, Color, Color],
  curtainState: { current: CurtainState | null },
) {
  const mix = (a: Color, b: Color, amount: number): Color => [
    a[0] + (b[0] - a[0]) * amount,
    a[1] + (b[1] - a[1]) * amount,
    a[2] + (b[2] - a[2]) * amount,
  ];
  const css = (color: Color, alpha = 1) =>
    `rgba(${color.map((value) => Math.round(value * 255)).join(",")},${alpha})`;
  sceneBackground(ctx, background, width, height);
  const dpr = Math.max(0.01, width / Math.max(1, window.innerWidth));
  ctx.save();
  ctx.scale(dpr, dpr);
  const w = width / dpr;
  const h = height / dpr;

  if (preset === "curtain") {
    const bass = reactive.bass;
    const mid = reactive.mid;
    const treble = reactive.treble;
    const cols = Math.max(10, Math.min(30, Math.floor(w / 62)));
    const rows = Math.max(5, Math.min(16, Math.floor(h / 74)));
    const state = curtainCellsFor(cols, rows, clock, curtainState.current);
    curtainState.current = state;
    const digitW = w / cols;
    const digitH = Math.min(digitW * 1.55, h / rows);
    const cellW = Math.min(digitW * 0.42, digitH * 0.34);
    const cellH = cellW * 1.18;
    const flashDecay = Math.exp(-delta * 6);
    if ((reactive.bassAttack ?? 0) > 0.05) {
      for (const cell of state.cells) if (Math.random() < 0.1) cell.flash = 1;
    }
    // A bigger section landing sweeps light across the whole curtain.
    if ((reactive.impact ?? 0) > 0.6)
      for (const cell of state.cells) if (Math.random() < 0.5) cell.flash = 1;
    for (const cell of state.cells) {
      if (cell.speedMult === 1 && Math.random() < mid * delta * 9)
        cell.speedMult = 18 + Math.random() * 20;
      if (clock - cell.lastChange > 1 / (cell.changeSpeed * cell.speedMult)) {
        cell.digit = Math.floor(Math.random() * 10);
        cell.lastChange = clock;
        if (cell.speedMult > 1) cell.speedMult = 1;
      }
      if (Math.random() < treble * delta * 7) cell.darken = 0;
      if (cell.darken < 1) cell.darken = Math.min(1, cell.darken + delta * 0.7);
      cell.flash *= flashDecay;
      const visible = SEGMENT_MAP[cell.digit];
      const brightness =
        Math.min(1, 0.05 + bass * 0.3 + (reactive.drive ?? 0) * 0.22 + cell.flash * 1.1) *
        cell.darken;
      const cellIndex = state.cells.indexOf(cell);
      const x = (cellIndex % cols) * digitW + (digitW - cellW * 2) / 2;
      const y = Math.floor(cellIndex / cols) * digitH + (digitH - (cellH * 2 + cellW * 0.36)) / 2;
      const segmentColor = (alpha: number) => css(mix(waves[1], waves[0], 0.3), alpha);
      ctx.lineWidth = Math.max(1.5, cellH * 0.22);
      ctx.lineCap = "round";
      const draw = (x1: number, y1: number, x2: number, y2: number, on: boolean) => {
        ctx.strokeStyle = on ? segmentColor(brightness) : segmentColor(0.025 * cell.darken);
        ctx.beginPath();
        ctx.moveTo(x + x1, y + y1);
        ctx.lineTo(x + x2, y + y2);
        ctx.stroke();
      };
      draw(cellW * 0.18, 0, cellW * 1.82, 0, !!visible[0]);
      draw(cellW * 2, cellH * 0.18, cellW * 2, cellH * 0.82, !!visible[1]);
      draw(cellW * 2, cellH * 1.18, cellW * 2, cellH * 1.82, !!visible[2]);
      draw(cellW * 0.18, cellH * 2, cellW * 1.82, cellH * 2, !!visible[3]);
      draw(0, cellH * 1.18, 0, cellH * 1.82, !!visible[4]);
      draw(0, cellH * 0.18, 0, cellH * 0.82, !!visible[5]);
      draw(cellW * 0.18, cellH, cellW * 1.82, cellH, !!visible[6]);
    }
  } else if (preset === "ascii") {
    const bass = reactive.bass;
    const mid = reactive.mid;
    const treble = reactive.treble;
    // Three tints: the two scene colours and their blend, chosen per glyph by the noise.
    const atlases = [
      tintedAtlas(0, waves[0], clock),
      tintedAtlas(1, mix(waves[0], waves[1], 0.5), clock),
      tintedAtlas(2, waves[1], clock),
    ];
    const cell = Math.max(22, Math.min(30, Math.floor(w / 34)));
    const cols = Math.ceil(w / cell);
    const rows = Math.ceil(h / cell);
    if (!asciiRandoms || asciiRandoms.length < cols * rows)
      asciiRandoms = Float32Array.from({ length: cols * rows }, () => Math.random());
    const rand = asciiRandoms;
    const noiseSpeed = 0.02 + mid * 0.3;
    const slowTime = Math.floor(clock * 9);
    ctx.font = `${cell * 0.92}px "DM Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < cols; column++) {
        const gridIndex = row * cols + column;
        const px = (column + 0.5) * cell;
        const py = (row + 0.5) * cell;
        const nx = (column / cols) * 3;
        const ny = (row / rows) * 3;
        const noise =
          (Math.sin(nx * 3 + clock * noiseSpeed * 9) +
            Math.sin(ny * 4 + clock * noiseSpeed * 4.5) +
            Math.sin(nx * 5 - clock * noiseSpeed * 1.8) +
            Math.sin(ny * 7 - clock * noiseSpeed * 7.2)) /
            8 +
          0.5;
        let charIndex = Math.floor((noise * 64 + rand[gridIndex] * 64 + slowTime * 7) % 64);
        if (rand[(gridIndex + slowTime) % rand.length] < treble * 0.3)
          charIndex = Math.floor(rand[(gridIndex * 7 + slowTime) % rand.length] * 64);
        const brightness = Math.max(
          0,
          Math.min(
            1,
            noise * 0.2 + bass * 0.4 + (reactive.drive ?? 0) * 0.3 + (reactive.impact ?? 0) * 0.5,
          ),
        );
        if (brightness < 0.3) continue;
        const edgeFade =
          Math.min(1, Math.sin(Math.PI * Math.min(1, ((row + 0.5) / rows) * 1.04)) * 0.7 + 0.3) *
          Math.min(1, Math.sin(Math.PI * Math.min(1, ((column + 0.5) / cols) * 1.04)) * 0.5 + 0.5);
        ctx.globalAlpha = Math.min(
          0.22 + (reactive.drive ?? 0) * 0.2 + (reactive.impact ?? 0) * 0.25,
          brightness * 0.45 * edgeFade,
        );
        const atlas = atlases[brightness > 0.85 ? 2 : Math.min(2, Math.floor(noise * 3))];
        ctx.drawImage(
          atlas,
          (charIndex % 8) * 32,
          Math.floor(charIndex / 8) * 32,
          32,
          32,
          px - cell / 2,
          py - cell / 2,
          cell,
          cell,
        );
      }
    }
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

export default function Backdrop() {
  const backdropRef = useRef<HTMLDivElement>(null);
  // Whether to draw below 1200 x 720 too (the viewer's choice); the loop restarts when it changes.
  const [smallScreenEffects, setSmallScreenEffects] = useState(false);
  useEffect(() => {
    const apply = (preferences: PlaybackPreferences) => {
      applySmallScreenEffects(preferences);
      setSmallScreenEffects(Boolean(preferences.smallScreenEffects));
      const value = preferences.backdropOpacity;
      const opacity =
        typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.5;
      backdropRef.current?.style.setProperty("--backdrop-opacity", String(opacity));
    };
    apply(readPlaybackPreferences());
    const update = (event: Event) => apply((event as CustomEvent<PlaybackPreferences>).detail);
    window.addEventListener("echora:playback-preferences", update);
    return () => window.removeEventListener("echora:playback-preferences", update);
  }, []);
  const glRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!smallScreenEffects && window.matchMedia(SMALL_SCREEN_QUERY).matches) return;
    const maybeGlCanvas = glRef.current;
    const maybeSceneCanvas = sceneRef.current;
    if (!maybeGlCanvas || !maybeSceneCanvas) return;
    const canvas: HTMLCanvasElement = maybeGlCanvas;
    const sceneCanvas: HTMLCanvasElement = maybeSceneCanvas;
    const scene = sceneCanvas.getContext("2d");
    if (!scene) return;
    let cancelled = false;
    let animation = 0;
    let removeResize = () => {};
    const target: Reactivity = { bass: 0, mid: 0, treble: 0, level: 0 };
    let preferences = readPlaybackPreferences();
    let paletteTarget: { background: Color; waves: [Color, Color, Color] } | null = null;
    const curtainState: { current: CurtainState | null } = { current: null };

    let opacityScale = 1;
    let targetOpacityScale = 1;
    const baseColor: Color = [...DEFAULT_BASE];
    const waveColors: [Color, Color, Color] = [
      [...DEFAULT_WAVES[0]],
      [...DEFAULT_WAVES[1]],
      [...DEFAULT_WAVES[2]],
    ];
    const reactive: Reactivity = { ...target };
    // One journey for the presets drawn here: waves, digital curtain and ASCII dance.
    const journey = new SongJourney();
    const receiveAudio = (event: Event) => {
      Object.assign(target, (event as CustomEvent<Reactivity>).detail);
      journey.receive((event as CustomEvent<VisualFrame>).detail, preferences);
    };
    const receiveMode = (event: Event) => {
      targetOpacityScale = (event as CustomEvent<{ waveOpacity: number }>).detail.waveOpacity;
    };
    const receivePreferences = (event: Event) => {
      preferences = (event as CustomEvent<PlaybackPreferences>).detail;
    };
    const receivePalette = (event: Event) => {
      const detail = (
        event as CustomEvent<{
          active: boolean;
          palette: { background: Color; waves: [Color, Color, Color] } | null;
        }>
      ).detail;
      paletteTarget = detail.active ? detail.palette : null;
    };
    window.addEventListener("echora:audio-reactivity", receiveAudio);
    window.addEventListener("echora:backdrop-mode", receiveMode);
    window.addEventListener("echora:track-palette", receivePalette);
    window.addEventListener("echora:playback-preferences", receivePreferences);

    async function start() {
      for (const file of scripts) await loadScript(file);
      if (cancelled) return;
      const gl = canvas.getContext("webgl2", {
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
      });
      const scope = window as XmbWindow;
      if (!gl || !scope.createSplineLayer || !scope.createParticlesLayer) return;
      gl.getExtension("OES_texture_float_linear");
      gl.getExtension("EXT_color_buffer_float");

      const resize = () => {
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        for (const [element, elementRatio] of [
          [canvas, ratio],
          [sceneCanvas, 1],
        ] as const) {
          const width = Math.max(1, Math.floor(element.clientWidth * elementRatio));
          const height = Math.max(1, Math.floor(element.clientHeight * elementRatio));
          if (element.width !== width || element.height !== height) {
            element.width = width;
            element.height = height;
          }
        }
        gl.viewport(0, 0, canvas.width, canvas.height);
      };
      window.addEventListener("resize", resize);
      removeResize = () => window.removeEventListener("resize", resize);
      resize();

      const spline = scope.createSplineLayer(gl, canvas);
      const particles = scope.createParticlesLayer(gl, canvas);
      const splineSettings = scope.SPLINE_SETTINGS;
      const particleSettings = scope.PARTICLE_SETTINGS;
      const baseline =
        splineSettings && particleSettings
          ? {
              spline: {
                layerAmplitudes: [...splineSettings.layerAmplitudes],
                opacity: splineSettings.opacity,
                layerColors: splineSettings.layerColors.map((color) => [...color] as Color),
                background: [
                  splineSettings.colorR,
                  splineSettings.colorG,
                  splineSettings.colorB,
                ] as Color,
              },
              particles: { count: particleSettings.count },
            }
          : null;
      const waveSprings = [0, 1, 2].map((layer) => ({
        position: baseline?.spline.layerAmplitudes[layer] ?? 0,
        velocity: 0,
      }));
      let previous = performance.now();
      let lastRendered = 0;
      let elapsed = 0;
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const frame = (now: number) => {
        const activeFrameInterval =
          preferences.waveFrameRate === "uncapped" ? 0 : 1000 / Number(preferences.waveFrameRate);
        const frameInterval = preferences.wavesEnabled ? activeFrameInterval : 500;
        if (now - lastRendered < frameInterval) {
          if (!reduced && !cancelled) animation = requestAnimationFrame(frame);
          return;
        }
        const animationSpeed =
          preferences.animationSpeed === "slow"
            ? 0.65
            : preferences.animationSpeed === "fast"
              ? 1.45
              : 1;
        const frameDelta = Math.min(0.1, Math.max(0.001, (now - previous) / 1000));
        journey.step(frameDelta, animationSpeed, now / 1000);
        // The waves flow faster as the song pushes harder, and surge through a build-up.
        if (preferences.wavesEnabled)
          elapsed +=
            (Math.max(0, now - previous) / 1000) *
            animationSpeed *
            (0.6 + journey.drive * 0.8 + journey.anticipation * 0.5);
        previous = now;
        lastRendered = now;
        for (const key of ["bass", "mid", "treble", "level"] as const)
          reactive[key] += (target[key] - reactive[key]) * 0.11;
        reactive.bassAttack = target.bassAttack ?? 0;
        reactive.midAttack = target.midAttack ?? 0;
        reactive.trebleAttack = target.trebleAttack ?? 0;
        reactive.onset = target.onset;
        reactive.timestamp = target.timestamp;
        target.onset = false;
        opacityScale += (targetOpacityScale - opacityScale) * 0.035;
        // Scenes recolour the waves from the artwork palette; the background lifts with the push.
        const lift = 0.85 + journey.drive * 0.35 + journey.impact * 0.35;
        const background = (paletteTarget?.background ?? DEFAULT_BASE).map((value) =>
          Math.min(255, value * lift),
        ) as Color;
        const artwork = paletteTarget?.waves ?? DEFAULT_WAVES;
        const colors = [
          journey.color(artwork, 0),
          journey.color(artwork, 1),
          journey.color(artwork, 2),
        ];
        for (let channel = 0; channel < 3; channel++)
          baseColor[channel] += (background[channel] - baseColor[channel]) * 0.025;
        for (let layer = 0; layer < 3; layer++) {
          for (let channel = 0; channel < 3; channel++)
            waveColors[layer][channel] +=
              (colors[layer][channel] - waveColors[layer][channel]) * 0.035;
        }
        const rootsActive = preferences.wavesEnabled && preferences.backdropPreset === "roots";
        const fallActive =
          preferences.wavesEnabled && preferences.backdropPreset === "lightningfall";
        const signalActive =
          preferences.backdropPreset === "oscilloscope" ||
          preferences.backdropPreset === "void" ||
          preferences.backdropPreset === "clouds" ||
          preferences.backdropPreset === "meshgrid" ||
          preferences.backdropPreset === "waterdrops";
        const sceneActive =
          preferences.wavesEnabled &&
          preferences.backdropPreset !== "waves" &&
          !rootsActive &&
          !fallActive &&
          !signalActive;
        sceneCanvas.style.opacity = sceneActive ? "1" : "0";
        canvas.style.opacity =
          sceneActive ||
          rootsActive ||
          fallActive ||
          signalActive ||
          (preferences.backdropPreset === "waves" && !preferences.wavesEnabled)
            ? "0"
            : "1";
        if (sceneActive && scene) {
          const sceneReactive: Reactivity = {
            ...reactive,
            bass: reactive.bass * preferences.bassReactivity,
            mid: reactive.mid * preferences.vocalReactivity,
            treble: reactive.treble * preferences.trebleReactivity,
            drive: journey.drive,
            impact: journey.impact,
          };
          presetScene(
            scene,
            preferences.backdropPreset,
            sceneCanvas.width,
            sceneCanvas.height,
            elapsed,
            now / 1000,
            frameDelta,
            sceneReactive,
            [...baseColor] as Color,
            [[...waveColors[0]], [...waveColors[1]], [...waveColors[2]]],
            curtainState,
          );
        } else if (
          !rootsActive &&
          !fallActive &&
          !signalActive &&
          baseline &&
          splineSettings &&
          particleSettings
        ) {
          splineSettings.opacity = baseline.spline.opacity * opacityScale;
          const waveLevels = [
            target.bass * preferences.bassReactivity,
            target.mid * preferences.vocalReactivity,
            target.treble * preferences.trebleReactivity,
          ];
          for (let layer = 0; layer < 3; layer++) {
            const swell =
              0.7 + journey.drive * 0.6 + journey.anticipation * 0.5 + journey.impact * 0.9;
            const desired =
              preferences.wavesEnabled && preferences.backdropPreset === "waves"
                ? (baseline.spline.layerAmplitudes[layer] +
                    waveLevels[layer] * [1.65, 1.3, 1.05][layer]) *
                  swell
                : 0;
            splineSettings.layerAmplitudes[layer] = advanceWaveSpring(
              waveSprings[layer],
              desired,
              frameDelta,
              [7, 8, 9][layer],
            );
          }
          particleSettings.count =
            preferences.wavesEnabled && preferences.backdropPreset === "waves"
              ? Math.round(
                  (baseline.particles.count +
                    reactive.treble * 2600 +
                    journey.drive * 1400 +
                    journey.impact * 2000) /
                    100,
                ) * 100
              : 0;
          splineSettings.colorR += (baseColor[0] - splineSettings.colorR) * 0.025;
          splineSettings.colorG += (baseColor[1] - splineSettings.colorG) * 0.025;
          splineSettings.colorB += (baseColor[2] - splineSettings.colorB) * 0.025;
          for (let layer = 0; layer < splineSettings.layerColors.length; layer++) {
            for (let channel = 0; channel < 3; channel++)
              splineSettings.layerColors[layer][channel] +=
                (waveColors[layer][channel] - splineSettings.layerColors[layer][channel]) * 0.035;
          }
          spline.render(elapsed);
          particles.render(elapsed + 417);
        }
        if (!reduced && !cancelled) animation = requestAnimationFrame(frame);
      };
      animation = requestAnimationFrame(frame);
    }

    start().catch(() => {});
    return () => {
      cancelled = true;
      cancelAnimationFrame(animation);
      removeResize();
      window.removeEventListener("echora:audio-reactivity", receiveAudio);
      window.removeEventListener("echora:backdrop-mode", receiveMode);
      window.removeEventListener("echora:track-palette", receivePalette);
      window.removeEventListener("echora:playback-preferences", receivePreferences);
    };
  }, [smallScreenEffects]);

  return (
    <div ref={backdropRef} className={styles.backdrop} aria-hidden="true">
      <canvas ref={glRef} />
      <canvas ref={sceneRef} className={styles.scene} />
      <RootVisualizer />
      <SignalVisualizer />
      <VoidTunnelVisualizer />
      <CloudVisualizer />
      <LightningFallVisualizer />
      <MeshGridVisualizer />
      <WaterDropVisualizer />
    </div>
  );
}
