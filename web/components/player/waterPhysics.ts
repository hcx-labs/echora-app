import type { VisualFrame } from "./visualFeatures";
import type { PlaybackPreferences } from "./playbackPreferences";

const unit = (value: number) => Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
export type WaterControls = { energy: number; rate: number; fallSpeed: number; gravity: number; radius: number; impact: number; tension: number; damping: number; attack: number };

export function waterControls(frame: VisualFrame, preferences: PlaybackPreferences): WaterControls {
  const bass = frame.active ? unit(frame.bass * preferences.bassReactivity) : 0;
  const mid = frame.active ? unit(frame.mid * preferences.vocalReactivity) : 0;
  const treble = frame.active ? unit(frame.treble * preferences.trebleReactivity) : 0;
  const energy = bass * .5 + mid * .3 + treble * .2;
  const attack = frame.active ? unit(frame.bassAttack * preferences.bassReactivity * 4 + frame.trebleAttack * preferences.trebleReactivity * 2) : 0;
  const tempo = Number.isFinite(frame.bpm) && frame.bpm ? Math.max(.5, Math.min(2, frame.bpm / 100)) : 1;
  return {
    energy, attack,
    rate: energy > .015 ? (.3 + energy * 3.5) * tempo : 0,
    fallSpeed: 3 + bass * 5 + mid * 10,
    gravity: 12 + energy * 18,
    radius: 1.4 + bass * 3.6 + mid,
    impact: .16 + bass * .65 + attack * .35,
    // The fixed-step 2D solver requires tension below .5 for stability.
    tension: .15 + mid * .16 + treble * .12,
    damping: .984 + mid * .009 - treble * .01,
  };
}

/** Damped height-field wave equation. Impacts propagate and interfere in the same grid. */
export class WaterSurface {
  readonly height: Float32Array;
  readonly velocity: Float32Array;
  private readonly next: Float32Array;
  private remainder = 0;
  readonly width: number;
  readonly depth: number;
  constructor(width = 240, depth = 150) {
    this.width = width; this.depth = depth;
    this.height = new Float32Array(width * depth);
    this.velocity = new Float32Array(width * depth);
    this.next = new Float32Array(width * depth);
  }
  clear() { this.height.fill(0); this.velocity.fill(0); this.next.fill(0); this.remainder = 0; }
  impact(u: number, v: number, radius: number, strength: number) {
    if (![u, v, radius, strength].every(Number.isFinite)) return;
    radius = Math.max(1, Math.min(8, radius));
    strength = Math.max(0, Math.min(1.5, strength));
    const x = unit(u) * (this.width - 1), y = unit(v) * (this.depth - 1);
    const range = Math.ceil(radius * 2);
    for (let row = Math.max(1, Math.floor(y) - range); row <= Math.min(this.depth - 2, Math.ceil(y) + range); row++) {
      for (let col = Math.max(1, Math.floor(x) - range); col <= Math.min(this.width - 2, Math.ceil(x) + range); col++) {
        const distance = ((col - x) ** 2 + (row - y) ** 2) / (radius * radius);
        this.velocity[row * this.width + col] -= Math.exp(-distance * 2) * strength;
      }
    }
  }
  advance(seconds: number, tension: number, damping: number) {
    if (!Number.isFinite(seconds) || seconds <= 0) return;
    // Fixed 180Hz physics preserves wave speed on the denser grid and is
    // independent of the selected display frame rate.
    this.remainder += Math.min(.1, seconds);
    const c = Math.max(.08, Math.min(.45, Number.isFinite(tension) ? tension : .15));
    const drag = Math.max(.96, Math.min(.995, Number.isFinite(damping) ? damping : .984));
    while (this.remainder + 1e-10 >= 1 / 180) {
      this.remainder -= 1 / 180;
      for (let y = 1; y < this.depth - 1; y++) {
        for (let x = 1; x < this.width - 1; x++) {
          const i = y * this.width + x, h = this.height[i];
          const laplacian = this.height[i - 1] + this.height[i + 1] + this.height[i - this.width] + this.height[i + this.width] - 4 * h;
          const edge = Math.min(x, y, this.width - 1 - x, this.depth - 1 - y);
          const absorb = edge < 8 ? .8 + edge * .025 : 1;
          const velocity = (this.velocity[i] + laplacian * c - h * .001) * drag * absorb;
          this.velocity[i] = velocity;
          this.next[i] = Math.max(-2, Math.min(2, h + velocity));
        }
      }
      this.height.set(this.next);
    }
  }
}
