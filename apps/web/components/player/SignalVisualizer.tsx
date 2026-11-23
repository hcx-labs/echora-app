"use client";

import type { VisualFrame } from "./visualFeatures";
import type { TrackPalette } from "./artworkPalette";
import * as THREE from "three";
import { useEffect, useRef } from "react";
import { readPlaybackPreferences, type PlaybackPreferences } from "./playbackPreferences";
import { SongJourney } from "./songJourney";
import styles from "./SignalVisualizer.module.css";

type Color = [number, number, number];
const SAMPLES = 256;
const fragmentShader = `
  uniform vec2 resolution;
  uniform sampler2D waveform;
  uniform vec3 colorA, colorB, colorC;
  uniform float travel, drive, bass, mid, treble, pulse, beatPhase, anticipation, impact, shift;
  uniform float opacity;
  varying vec2 vUv;
  const float PI = 3.14159265;
  void main() {
    vec2 p = vUv - .5;
    p.x *= resolution.x / resolution.y;
    vec3 rgb = vec3(0.0);
    {
      // Oscilloscope: the trace takes the scene colours from left to right and
      // leaves fading echoes when a bigger section lands.
      float x = vUv.x;
      float sampleValue = texture2D(waveform, vec2(x, .5)).r * 2.0 - 1.0;
      float height = .1 + drive * .1 + anticipation * .05;
      vec3 trace = mix(colorA, colorB, smoothstep(.1, .9, x));
      float ends = smoothstep(0.0, .12, x) * smoothstep(0.0, .12, 1.0 - x);
      for (int echo = 0; echo < 3; echo++) {
        float e = float(echo);
        float offset = e * (.035 + impact * .05);
        float y = sampleValue * height * (1.0 - e * .18);
        float d = min(abs(vUv.y - .5 - y - offset), abs(vUv.y - .5 - y + offset));
        float aa = max(fwidth(vUv.y - y), 1.0 / resolution.y);
        float core = 1.0 - smoothstep(0.0, aa * (1.3 + drive), d);
        float halo = exp(-d * resolution.y * (.2 - drive * .08));
        float weight = e < .5 ? 1.0 : (.22 + impact * .5) * (1.0 - e * .3);
        rgb += mix(trace, colorC, e * .35) * (core * .55 + halo * (.14 + pulse * .12)) * weight;
      }
      // A faint mirrored floor reflection.
      float mirror = abs(vUv.y - .5 + sampleValue * height * .6 + .26);
      rgb += trace * exp(-mirror * resolution.y * .08) * .05 * (.4 + drive);
      rgb *= ends * (.55 + drive * .55);
    }
    // Soft roll-off keeps hue when lights overlap.
    rgb = rgb / (1.0 + rgb * .6);
    gl_FragColor = vec4(rgb, opacity);
  }
`;

/** One GPU draw call for the oscilloscope, moving through the song's journey. */
export default function SignalVisualizer() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const compact = window.matchMedia("(max-width:1199px), (max-height:719px)");
    let preferences = readPlaybackPreferences();
    let renderer: THREE.WebGLRenderer | null = null;
    const scene = new THREE.Scene();
    const camera = new THREE.Camera();
    const samples = new Uint8Array(SAMPLES).fill(128);
    const texture = new THREE.DataTexture(samples, SAMPLES, 1, THREE.RedFormat);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const geometry = new THREE.PlaneGeometry(2, 2);
    const uniforms = {
      waveform: { value: texture },
      resolution: { value: new THREE.Vector2(1, 1) },
      colorA: { value: new THREE.Vector3(0.3, 0.7, 0.65) },
      colorB: { value: new THREE.Vector3(0.5, 0.4, 0.8) },
      colorC: { value: new THREE.Vector3(0.6, 0.8, 1) },
      travel: { value: 0 },
      drive: { value: 0 },
      opacity: { value: 1 },
      bass: { value: 0 },
      mid: { value: 0 },
      treble: { value: 0 },
      pulse: { value: 0 },
      beatPhase: { value: 0 },
      anticipation: { value: 0 },
      impact: { value: 0 },
      shift: { value: 0 },
    };
    const material = new THREE.ShaderMaterial({
      depthTest: false,
      depthWrite: false,
      uniforms,
      vertexShader: "varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}",
      fragmentShader,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    const journey = new SongJourney();
    let palette: Color[] = [
      [0.3, 0.7, 0.65],
      [0.5, 0.4, 0.8],
      [0.6, 0.8, 1],
    ];
    let frame = 0,
      lastFrame = 0,
      active = false,
      failed = false;
    let targetBass = 0,
      targetMid = 0,
      targetTreble = 0,
      lastAudio = -Infinity;
    const resize = () => {
      if (!renderer) return;
      const bounds = canvas.getBoundingClientRect();
      renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
      renderer.setSize(Math.max(1, bounds.width), Math.max(1, bounds.height), false);
      renderer.getDrawingBufferSize(uniforms.resolution.value);
    };
    const draw = (now: number) => {
      if (!active || !renderer) return;
      frame = requestAnimationFrame(draw);
      const interval =
        preferences.waveFrameRate === "uncapped" ? 0 : 1000 / Number(preferences.waveFrameRate);
      if (now - lastFrame < interval) return;
      const dt = Math.min(0.05, (now - lastFrame) / 1000 || 0.016);
      lastFrame = now;
      const audible = now - lastAudio < 700;
      const rate =
        preferences.animationSpeed === "slow"
          ? 0.65
          : preferences.animationSpeed === "fast"
            ? 1.45
            : 1;
      journey.step(dt, rate, now / 1000);
      for (const [name, target] of [
        ["bass", targetBass],
        ["mid", targetMid],
        ["treble", targetTreble],
      ] as const) {
        const current = uniforms[name].value,
          next = audible ? target : 0;
        // Fast attack preserves beats; slower release avoids strobing.
        uniforms[name].value += (next - current) * (1 - Math.exp(-dt * (next > current ? 18 : 5)));
      }
      uniforms.travel.value = journey.travel * 1.6;
      uniforms.drive.value = journey.drive;
      uniforms.pulse.value = journey.beatPulse * preferences.bassReactivity;
      uniforms.beatPhase.value = journey.beatPhase;
      uniforms.anticipation.value = journey.anticipation;
      uniforms.impact.value = journey.impact;
      uniforms.shift.value = journey.shift;
      for (const [name, slot] of [
        ["colorA", 0],
        ["colorB", 1],
        ["colorC", 2],
      ] as const) {
        const color = journey.color(palette, slot);
        uniforms[name].value.lerp(new THREE.Vector3(...color), 1 - Math.exp(-dt * 3));
      }
      if (!audible) {
        for (let i = 0; i < SAMPLES; i++)
          samples[i] = Math.round(samples[i] + (128 - samples[i]) * Math.min(1, dt * 8));
        texture.needsUpdate = true;
      }
      renderer.render(scene, camera);
    };
    const sync = () => {
      const next =
        !failed &&
        preferences.wavesEnabled &&
        preferences.backdropPreset === "oscilloscope" &&
        !compact.matches &&
        !reduced.matches &&
        !document.hidden;
      canvas.style.display = next ? "block" : "none";
      if (next && !renderer) {
        try {
          renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
          resize();
        } catch {
          failed = true;
          canvas.style.display = "none";
          return;
        }
      }
      if (next === active) return;
      active = next;
      cancelAnimationFrame(frame);
      if (active) {
        lastFrame = performance.now();
        frame = requestAnimationFrame(draw);
      }
    };
    const receivePreferences = (event: Event) => {
      preferences = (event as CustomEvent<PlaybackPreferences>).detail;
      sync();
    };
    const receiveAudio = (event: Event) => {
      const detail = (event as CustomEvent<VisualFrame>).detail;
      journey.receive(detail, preferences);
      if (!detail.active) return;
      lastAudio = performance.now();
      targetBass = Math.min(1, detail.bass * preferences.bassReactivity * 1.4);
      targetMid = Math.min(1, detail.mid * preferences.vocalReactivity * 1.5);
      targetTreble = Math.min(1, detail.treble * preferences.trebleReactivity * 1.8);
      if (!active || preferences.backdropPreset !== "oscilloscope") return;
      const bins = detail.waveform;
      for (let i = 0; i < SAMPLES; i++) {
        const start = Math.floor((i * bins.length) / SAMPLES);
        const end = Math.min(
          bins.length,
          Math.max(start + 1, Math.floor(((i + 1) * bins.length) / SAMPLES)),
        );
        let sum = 0;
        for (let j = start; j < end; j++) sum += bins[j];
        samples[i] = end > start ? Math.round((sum / (end - start) + 1) * 127.5) : 128;
      }
      texture.needsUpdate = true;
    };
    const receivePalette = (event: Event) => {
      const { palette: next } = (event as CustomEvent<{ palette: TrackPalette | null }>).detail;
      if (next) palette = next.trails?.length >= 3 ? next.trails : next.waves;
    };
    const reset = () => {
      samples.fill(128);
      texture.needsUpdate = true;
      targetBass = targetMid = targetTreble = 0;
      lastAudio = -Infinity;
      journey.reset();
      for (const name of [
        "drive",
        "bass",
        "mid",
        "treble",
        "pulse",
        "impact",
        "anticipation",
      ] as const)
        uniforms[name].value = 0;
    };
    const events: [string, EventListener][] = [
      ["echora:playback-preferences", receivePreferences],
      ["echora:visual-frame", receiveAudio],
      ["echora:track-palette", receivePalette],
      ["echora:track-change", reset],
      ["resize", resize],
    ];
    events.forEach(([name, handler]) => window.addEventListener(name, handler));
    compact.addEventListener("change", sync);
    reduced.addEventListener("change", sync);
    document.addEventListener("visibilitychange", sync);
    sync();
    return () => {
      cancelAnimationFrame(frame);
      events.forEach(([name, handler]) => window.removeEventListener(name, handler));
      compact.removeEventListener("change", sync);
      reduced.removeEventListener("change", sync);
      document.removeEventListener("visibilitychange", sync);
      geometry.dispose();
      material.dispose();
      texture.dispose();
      renderer?.dispose();
    };
  }, []);
  return <canvas ref={ref} className={styles.signal} aria-hidden="true" />;
}
