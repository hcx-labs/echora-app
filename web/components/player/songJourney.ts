/**
 * The song as a journey: sections planned ahead from the cached self-similarity
 * structure, and a live state every backdrop reads so they move through the song
 * together instead of only twitching with the current loudness.
 *
 * Sections are planned once per track in visualFeatures.ts (planSongSections) and
 * arrive on every frame; a chorus that returns comes back to the same scene.
 *
 * Live state (per frame): a build-up that rises before a section with more
 * energy, an impact when it lands, scene blends, a beat phase locked to the
 * detected beats, and a harmony drift measured against the song's home key.
 */

import type { FrameSection } from "./visualFeatures";

type Color = [number, number, number];

// OKLab, for hue rotation that keeps perceived lightness.
const toLinear = (n: number) => (n <= 0.04045 ? n / 12.92 : ((n + 0.055) / 1.055) ** 2.4);
const toEncoded = (n: number) =>
  n <= 0.0031308 ? 12.92 * n : 1.055 * Math.max(0, n) ** (1 / 2.4) - 0.055;
function oklab([red, green, blue]: Color): Color {
  const [r, g, b] = [red, green, blue].map(toLinear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}
function fromOklab([light, a, b]: Color): Color {
  const l = (light + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (light - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (light - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ].map((n) => Math.max(0, Math.min(1, toEncoded(Math.max(0, Math.min(1, n)))))) as Color;
}
/** Rotate hue and scale chroma/lightness in OKLab. Inputs and outputs are sRGB 0..1. */
export function gradeColor(color: Color, hue: number, chroma = 1, light = 1): Color {
  const [l, a, b] = oklab(color);
  const cos = Math.cos(hue),
    sin = Math.sin(hue);
  return fromOklab([
    Math.min(0.97, l * light),
    (a * cos - b * sin) * chroma,
    (a * sin + b * cos) * chroma,
  ]);
}

type Sensitivity = { bassReactivity: number; vocalReactivity: number; trebleReactivity: number };
type JourneyFrame = {
  active: boolean;
  trackId: string | null;
  timestamp: number;
  bpm: number | null;
  beat: boolean;
  bass: number;
  mid: number;
  treble: number;
  section?: FrameSection | null;
  features: { chroma: number[] } | null;
  source: { durationSeconds: number } | null;
  enrichment: { vocalActivation: number | null } | null;
};
const clamp01 = (n: number) => (Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0);
const approach = (value: number, target: number, rate: number, dt: number) =>
  value + (target - value) * (1 - Math.exp(-dt * rate));

/** One per visualizer. Feed every visual frame to receive(); call step() once per drawn frame. */
export class SongJourney {
  /** Integrated distance travelled; tempo sets the pace, the journey sets the push. */
  travel = 0;
  /** How hard the music is pushing right now, 0..1, slow enough to feel like momentum. */
  drive = 0;
  /** Loudness relative to this song's loudest moments so far, 0..1. */
  intensity = 0;
  /** Rises over the bars before a section with more energy, 0..1. */
  anticipation = 0;
  /** Lands when a bigger section starts, then decays over about a second. */
  impact = 0;
  /** Any section boundary, decays over about a second. */
  shift = 0;
  /** 0..1 position between beats, and a short pulse on each beat. */
  beatPhase = 0;
  beatPulse = 0;
  /** Beats counted this track; bar = Math.floor(beats / 4). */
  beats = 0;
  /** Vocal activation where the analysis has it, else 0. */
  voice = 0;
  progress = 0;
  /** Current and previous scene, blended over a few seconds after a change. */
  scene = 0;
  previousScene = 0;
  sceneBlend = 1;
  /** Energy of the current section within the song, 0..1. */
  sectionEnergy = 0.5;
  /** Hue offset in radians: how far the harmony has wandered from the song's home key. */
  harmonyShift = 0;
  bpm = 0;

  private trackId: string | null = null;
  private lastActive = -Infinity;
  private peak = 0.2;
  private target = { intensity: 0, anticipation: 0, harmonyX: 0, harmonyY: 0, voice: 0 };
  private home = { x: 0, y: 0 };
  private now = { x: 0, y: 0 };
  private sectionIndex = -1;
  private lastTimestamp: number | null = null;

  receive(frame: JourneyFrame, sensitivity: Sensitivity, nowSeconds = performance.now() / 1000) {
    if (frame.trackId !== this.trackId && frame.active) {
      this.reset();
      this.trackId = frame.trackId;
    }
    if (!frame.active) return;
    this.lastActive = nowSeconds;
    const energy = clamp01(
      frame.bass * sensitivity.bassReactivity * 0.5 +
        frame.mid * sensitivity.vocalReactivity * 0.35 +
        frame.treble * sensitivity.trebleReactivity * 0.15,
    );
    this.peak = Math.max(this.peak * 0.99995, energy, 0.08);
    this.target.intensity = clamp01(energy / this.peak);
    this.bpm = frame.bpm ?? this.bpm;
    if (frame.beat) {
      this.beatPulse = 1;
      this.beatPhase = 0;
      this.beats += 1;
    }
    this.target.voice = frame.enrichment?.vocalActivation ?? 0;
    if (frame.source) this.progress = clamp01(frame.timestamp / frame.source.durationSeconds);

    const continuous =
      this.lastTimestamp !== null &&
      frame.timestamp >= this.lastTimestamp &&
      frame.timestamp - this.lastTimestamp < 0.5;
    this.lastTimestamp = frame.timestamp;
    const section = frame.section;
    if (section) {
      if (section.index !== this.sectionIndex) {
        const rise = section.energy - this.sectionEnergy;
        // Seeks change scene quietly; only played-through boundaries hit.
        if (continuous && this.sectionIndex >= 0) {
          this.shift = 1;
          if (rise > 0.12) this.impact = Math.min(1, 0.45 + rise * 1.2);
        }
        if (section.scene !== this.scene) {
          this.previousScene = this.scene;
          this.scene = section.scene;
          this.sceneBlend = continuous ? 0 : 1;
        }
        this.sectionIndex = section.index;
        this.sectionEnergy = section.energy;
      }
      const rise = (section.nextEnergy ?? section.energy) - section.energy;
      const beat = 60 / Math.max(60, Math.min(200, this.bpm || 110));
      const window = Math.max(3, Math.min(9, beat * 16));
      const remaining = section.end - frame.timestamp;
      this.target.anticipation =
        rise > 0.15 && remaining < window
          ? clamp01((1 - remaining / window) ** 2 * Math.min(1, rise * 1.6))
          : 0;
    }

    // Circle of fifths: neighbouring keys sit close, so modulations drift the hue a little.
    const chroma = frame.features?.chroma;
    if (chroma) {
      let x = 0,
        y = 0;
      for (let pitch = 0; pitch < 12; pitch++) {
        const angle = (((pitch * 7) % 12) / 12) * Math.PI * 2;
        x += Math.cos(angle) * chroma[pitch];
        y += Math.sin(angle) * chroma[pitch];
      }
      this.target.harmonyX = x;
      this.target.harmonyY = y;
    }
  }

  step(dt: number, rate = 1, nowSeconds = performance.now() / 1000) {
    dt = Math.max(0, Math.min(0.1, dt));
    const audible = nowSeconds - this.lastActive < 0.7;
    this.intensity = approach(
      this.intensity,
      audible ? this.target.intensity : 0,
      this.target.intensity > this.intensity ? 6 : 1.4,
      dt,
    );
    this.anticipation = approach(this.anticipation, audible ? this.target.anticipation : 0, 3, dt);
    this.voice = approach(this.voice, audible ? this.target.voice : 0, 2, dt);
    const drive = audible
      ? clamp01(
          0.18 +
            this.sectionEnergy * 0.42 +
            this.intensity * 0.32 +
            this.anticipation * 0.3 +
            this.impact * 0.25,
        )
      : 0;
    this.drive = approach(this.drive, drive, drive > this.drive ? 2.2 : 1.1, dt);
    this.impact *= Math.exp(-dt * 1.5);
    this.shift *= Math.exp(-dt * 1.2);
    this.beatPulse *= Math.exp(-dt * 7);
    if (audible && this.bpm > 0)
      this.beatPhase = Math.min(1, this.beatPhase + (dt * this.bpm) / 60);
    this.sceneBlend = Math.min(1, this.sceneBlend + dt / 2.5);
    // Home key: a slow average of the harmony; the live harmony is faster.
    this.now.x = approach(this.now.x, this.target.harmonyX, 0.8, dt);
    this.now.y = approach(this.now.y, this.target.harmonyY, 0.8, dt);
    if (audible) {
      this.home.x = approach(this.home.x, this.target.harmonyX, 0.03, dt);
      this.home.y = approach(this.home.y, this.target.harmonyY, 0.03, dt);
    }
    const wander = Math.atan2(
      this.home.x * this.now.y - this.home.y * this.now.x,
      this.home.x * this.now.x + this.home.y * this.now.y,
    );
    const confidence = clamp01(Math.hypot(this.now.x, this.now.y) * 1.5);
    this.harmonyShift = approach(
      this.harmonyShift,
      Math.max(-0.5, Math.min(0.5, wander * 0.45)) * confidence,
      1,
      dt,
    );
    const tempo = Math.max(0.6, Math.min(1.8, (this.bpm || 100) / 100));
    if (audible)
      this.travel +=
        dt *
        rate *
        tempo *
        (0.12 + this.drive * 0.88 + this.impact * 0.7 + this.anticipation * 0.4);
  }

  /** Palette colour for a slot in the current scene: scenes rotate through the
   * artwork's colours, harmony nudges the hue, and energy lifts saturation and light. */
  color(palette: Color[], slot = 0): Color {
    if (!palette.length) return [0.5, 0.5, 0.5];
    const pick = (scene: number) => palette[(slot + scene) % palette.length];
    const from = pick(this.previousScene),
      to = pick(this.scene);
    const t = this.sceneBlend * this.sceneBlend * (3 - 2 * this.sceneBlend);
    const base = from.map((value, channel) => value + (to[channel] - value) * t) as Color;
    return gradeColor(
      base,
      this.harmonyShift,
      0.9 + this.drive * 0.35 + this.impact * 0.2,
      0.9 + this.drive * 0.16 + this.impact * 0.18,
    );
  }

  reset() {
    this.travel = this.drive = this.intensity = this.anticipation = this.impact = this.shift = 0;
    this.beatPhase =
      this.beatPulse =
      this.beats =
      this.voice =
      this.progress =
      this.harmonyShift =
        0;
    this.scene = this.previousScene = 0;
    this.sceneBlend = 1;
    this.sectionEnergy = 0.5;
    this.bpm = 0;
    this.peak = 0.2;
    this.sectionIndex = -1;
    this.lastTimestamp = null;
    this.lastActive = -Infinity;
    this.target = { intensity: 0, anticipation: 0, harmonyX: 0, harmonyY: 0, voice: 0 };
    this.home = { x: 0, y: 0 };
    this.now = { x: 0, y: 0 };
  }
}
