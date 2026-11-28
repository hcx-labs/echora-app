"use client";

import type { VisualFrame } from "./visualFeatures";
import type { TrackPalette } from "./artworkPalette";
import * as THREE from "three";
import { useEffect, useRef } from "react";
import { readPlaybackPreferences, type PlaybackPreferences } from "./playbackPreferences";
import { SMALL_SCREEN_QUERY, effectsHiddenBySize } from "./smallScreen";
import { SongJourney } from "./songJourney";
import { TunnelMotion } from "./tunnelMotion";
import styles from "./SignalVisualizer.module.css";

type Color = [number, number, number];
const BANDS = 24;
// Cross-sections a scene can take: 0 is a circle, positive values are polygons with that many
// sides, negative values are blooms with that many petals.
const SHAPES = [0, 6, -4, 8, -6, 5];
const TWISTS = [0, 0.22, -0.16, 0.34, -0.26, 0.1];

const fragmentShader = `
  uniform vec2 resolution;
  uniform sampler2D bands;
  uniform vec3 colorA, colorB, colorC;
  uniform vec2 bend;
  uniform float travel, rotation, spacing, beatFrac, kick, drive, treble, anticipation, impact;
  uniform float sidesFrom, sidesTo, shapeBlend, twist, progress, opacity;
  varying vec2 vUv;
  const float PI = 3.14159265;
  // Rings cross this radius on the beat.
  const float REF = .62;
  float polygon(float angle, float sides) {
    float n = abs(sides);
    if (n < 2.5) return 1.0;
    float sector = 2.0 * PI / n;
    float edge = cos(PI / n) / cos(mod(angle + PI / 2.0, sector) - sector * .5);
    return sides > 0.0 ? 1.0 / edge : edge;
  }
  void main() {
    vec2 p = vUv - .5;
    p.x *= resolution.x / resolution.y;
    float screen = length(p);
    // The far end of the tunnel leans along its path; the mouth stays put.
    vec2 q = p - bend * (1.0 - smoothstep(0.0, .75, screen));
    float radius = length(q * vec2(1.0, 1.1));
    float angle = atan(q.y, q.x);
    float far = -log(max(radius, .02));
    float turned = angle + rotation + far * twist;
    float shape = mix(polygon(turned, sidesFrom), polygon(turned, sidesTo), shapeBlend);
    float phase = .5 + log(REF) * spacing;
    // Which ring this pixel is nearest, and how many rings away from the viewer its centre is.
    float rough = -log(max(radius * shape, .02)) * spacing + travel + phase;
    float id = floor(rough);
    float ringAway = id + .5 - travel - phase;
    // Each ring shows one frequency band: bass at the mouth, treble deep in the tunnel.
    float band = clamp((ringAway - .3) / 8.0, 0.0, 1.0);
    float level = texture2D(bands, vec2((band * 23.0 + .5) / 24.0, .5)).r;
    // Low bands bend a ring in a few broad waves, high bands in many fine ones.
    float waves = floor(mix(2.0, 13.0, band));
    float wobble = sin(turned * waves + travel * .8 + id * 1.7) * level * mix(.06, .014, band);
    float depth = -log(max(radius * shape * (1.0 - wobble), .02)) * spacing + travel + phase;
    float ring = abs(fract(depth) - .5);
    float aa = max(fwidth(depth), .006);
    float width = .005 + level * .022;
    float line = 1.0 - smoothstep(width, width + aa * 1.2, ring);
    float halo = exp(-ring * (30.0 - level * 16.0));
    vec3 tint = band < .5 ? mix(colorA, colorB, band * 2.0) : mix(colorB, colorC, band * 2.0 - 1.0);
    // The ring that just reached the reference radius carries the beat.
    float reach = -log(REF) * spacing;
    float accent = exp(-beatFrac * 4.0) * kick * (1.0 - smoothstep(.3, 1.2, abs(ringAway - reach)));
    // Deep rings are small, so they get more light to stay readable.
    float light = (.08 + drive * .2 + level * (.5 + drive * .45) + accent * .9) * (1.0 + impact * .4) * (1.0 + band * .8);
    vec3 rgb = tint * (line * light + halo * (.025 + level * .13));
    // Wall panels between rings, lit by the treble.
    float panels = pow(.5 + .5 * cos(turned * max(3.0, abs(sidesTo)) + id * 2.1), 6.0);
    rgb += colorC * panels * halo * (.02 + treble * .12) * (.5 + drive);
    // Warp streaks while a build-up gathers.
    float streakAngle = fract(angle / (2.0 * PI) * 48.0 + floor(depth * .5) * .37);
    float streak = exp(-abs(streakAngle - .5) * 40.0) * smoothstep(.12, .5, radius) * fract(depth * .5);
    rgb += mix(colorC, vec3(1.0), .35) * streak * anticipation * .5;
    rgb *= smoothstep(.03, .16, radius) * (1.0 - smoothstep(.8, 1.35, screen));
    // A far light that grows as the song goes on: the end of the journey.
    rgb += colorC * exp(-radius * radius * 90.0) * (.05 + progress * .12) * (.6 + drive * .4);
    rgb = rgb / (1.0 + rgb * .6);
    gl_FragColor = vec4(rgb, opacity);
  }
`;

/** Per-band level relative to that band's own recent range, so every ring has contrast. */
class BandLevels {
  readonly values = new Float32Array(BANDS);
  private floor = new Float32Array(BANDS).fill(1);
  private peak = new Float32Array(BANDS);
  private target = new Float32Array(BANDS);
  private eased = new Float32Array(BANDS);

  receive(bands: number[], gain: (band: number) => number) {
    for (let i = 0; i < BANDS; i++) {
      const v = bands[i] ?? 0;
      this.floor[i] = v < this.floor[i] ? v : this.floor[i] + (v - this.floor[i]) * 0.0015;
      this.peak[i] = v > this.peak[i] ? v : this.peak[i] + (v - this.peak[i]) * 0.003;
      const range = Math.max(0.08, this.peak[i] - this.floor[i]);
      const level = Math.max(0, Math.min(1, (v - this.floor[i]) / range));
      this.target[i] = Math.min(1, level ** 1.2 * gain(i));
    }
  }

  step(dt: number, audible: boolean, scale: number) {
    for (let i = 0; i < BANDS; i++) {
      // Two smoothing stages: the first takes the frame-to-frame flicker out, the second eases the
      // ring into its new level (rising in about a quarter second, settling over about a second).
      const next = audible ? this.target[i] * scale : 0;
      this.eased[i] += (next - this.eased[i]) * (1 - Math.exp(-dt * 10));
      const rate = this.eased[i] > this.values[i] ? 4.5 : 1.8;
      this.values[i] += (this.eased[i] - this.values[i]) * (1 - Math.exp(-dt * rate));
    }
  }

  reset() {
    this.values.fill(0);
    this.target.fill(0);
    this.eased.fill(0);
    this.floor.fill(1);
    this.peak.fill(0);
  }
}

/** The void tunnel: a flight through the song, rings on the beat grid, one frequency band each. */
export default function VoidTunnelVisualizer() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const compact = window.matchMedia(SMALL_SCREEN_QUERY);
    let preferences = readPlaybackPreferences();
    let renderer: THREE.WebGLRenderer | null = null;
    const scene = new THREE.Scene();
    const camera = new THREE.Camera();
    const pixels = new Uint8Array(BANDS);
    const texture = new THREE.DataTexture(pixels, BANDS, 1, THREE.RedFormat);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const geometry = new THREE.PlaneGeometry(2, 2);
    const uniforms = {
      bands: { value: texture },
      resolution: { value: new THREE.Vector2(1, 1) },
      colorA: { value: new THREE.Vector3(0.3, 0.7, 0.65) },
      colorB: { value: new THREE.Vector3(0.5, 0.4, 0.8) },
      colorC: { value: new THREE.Vector3(0.6, 0.8, 1) },
      bend: { value: new THREE.Vector2(0, 0) },
      travel: { value: 0 },
      rotation: { value: 0 },
      spacing: { value: 3.3 },
      beatFrac: { value: 1 },
      kick: { value: 0 },
      drive: { value: 0 },
      treble: { value: 0 },
      anticipation: { value: 0 },
      impact: { value: 0 },
      progress: { value: 0 },
      sidesFrom: { value: 0 },
      sidesTo: { value: 0 },
      shapeBlend: { value: 1 },
      twist: { value: 0 },
      opacity: { value: 1 },
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
    const motion = new TunnelMotion();
    const levels = new BandLevels();
    let palette: Color[] = [
      [0.3, 0.7, 0.65],
      [0.5, 0.4, 0.8],
      [0.6, 0.8, 1],
    ];
    let frame = 0,
      lastFrame = 0,
      active = false,
      failed = false,
      lastAudio = -Infinity,
      section = 0,
      bendAmount = 0;
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
      const seconds = now / 1000;
      const audible = now - lastAudio < 700;
      const rate =
        preferences.animationSpeed === "slow"
          ? 0.65
          : preferences.animationSpeed === "fast"
            ? 1.45
            : 1;
      journey.step(dt, rate, seconds);
      const blend = journey.sceneBlend * journey.sceneBlend * (3 - 2 * journey.sceneBlend);
      const sidesTo = SHAPES[journey.scene % SHAPES.length];
      motion.step(dt, seconds, {
        audible,
        energy: journey.sectionEnergy,
        anticipation: journey.anticipation,
        sides: sidesTo,
        section,
        bpm: journey.bpm,
        rate,
      });
      levels.step(dt, audible, 0.45 + journey.drive * 0.75);
      for (let i = 0; i < BANDS; i++) pixels[i] = Math.round(Math.min(1, levels.values[i]) * 255);
      texture.needsUpdate = true;
      const values = levels.values;
      uniforms.kick.value = (values[0] + values[1] + values[2]) / 3;
      let high = 0;
      for (let i = 16; i < BANDS; i++) high += values[i];
      uniforms.treble.value = high / (BANDS - 16);
      uniforms.travel.value = motion.travel;
      uniforms.rotation.value = motion.rotation;
      uniforms.beatFrac.value = motion.beat - Math.floor(motion.beat);
      uniforms.drive.value = journey.drive;
      uniforms.anticipation.value = journey.anticipation;
      uniforms.impact.value = journey.impact;
      uniforms.progress.value = journey.progress;
      uniforms.sidesFrom.value = SHAPES[journey.previousScene % SHAPES.length];
      uniforms.sidesTo.value = sidesTo;
      uniforms.shapeBlend.value = blend;
      const twistFrom = TWISTS[journey.previousScene % TWISTS.length];
      uniforms.twist.value =
        twistFrom + (TWISTS[journey.scene % TWISTS.length] - twistFrom) * blend;
      // The path winds slowly with distance travelled; bigger sections take wider curves.
      bendAmount += (0.04 + journey.sectionEnergy * 0.08 - bendAmount) * (1 - Math.exp(-dt * 0.5));
      const t = motion.travel;
      uniforms.bend.value.set(
        (Math.sin(t * 0.047) * 0.7 + Math.sin(t * 0.019 + 1.3) * 0.3) * bendAmount,
        (Math.cos(t * 0.036) * 0.6 + Math.sin(t * 0.013 + 0.4) * 0.4) * bendAmount * 0.7,
      );
      for (const [name, slot] of [
        ["colorA", 0],
        ["colorB", 1],
        ["colorC", 2],
      ] as const) {
        const color = journey.color(palette, slot);
        uniforms[name].value.lerp(new THREE.Vector3(...color), 1 - Math.exp(-dt * 3));
      }
      renderer.render(scene, camera);
    };
    const sync = () => {
      const next =
        !failed &&
        preferences.wavesEnabled &&
        preferences.backdropPreset === "void" &&
        !effectsHiddenBySize(compact, preferences) &&
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
      const now = performance.now();
      lastAudio = now;
      motion.receive(detail.beatClock ?? null, now / 1000);
      if (detail.section) section = detail.section.index;
      levels.receive(detail.bands, (band) =>
        band < 8
          ? preferences.bassReactivity
          : band < 16
            ? preferences.vocalReactivity
            : preferences.trebleReactivity,
      );
    };
    const receivePalette = (event: Event) => {
      const { palette: next } = (event as CustomEvent<{ palette: TrackPalette | null }>).detail;
      if (next) palette = next.trails?.length >= 3 ? next.trails : next.waves;
    };
    const reset = () => {
      lastAudio = -Infinity;
      section = 0;
      journey.reset();
      motion.reset();
      levels.reset();
      pixels.fill(0);
      texture.needsUpdate = true;
      for (const name of ["drive", "kick", "treble", "impact", "anticipation"] as const)
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
