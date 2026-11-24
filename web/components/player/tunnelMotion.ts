/**
 * Motion for the void tunnel, locked to the song's beat grid.
 *
 * travel counts rings: a ring crosses the tunnel's reference radius exactly when travel is a
 * whole number, and travel advances by `gear` rings per beat, so rings land on the beats (gear 1),
 * every other beat (0.5) or on beats and off-beats (2). The gear follows the section's energy and
 * changes only on a bar line, where every gear is whole, so the rings stay on the beat across a
 * change. Because the clock is the beat grid itself, a song that speeds up or slows down moves the
 * tunnel with it.
 *
 * rotation turns half a sector of the tunnel's cross-section per bar, mostly on the downbeat in
 * energetic sections and evenly through the bar in calm ones; each section turns the other way.
 */

import type { BeatClock } from "./visualFeatures";

export type TunnelStep = {
  audible: boolean;
  /** Energy of the current section within the song, 0..1. */
  energy: number;
  /** Build-up before a bigger section, 0..1: shifts up a gear early. */
  anticipation: number;
  /** Sides of the cross-section (0 for a circle). */
  sides: number;
  section: number;
  /** Fallback when the song has no beat grid. */
  bpm: number;
  /** Animation speed preference: below 1 shifts down a gear, above 1 up. */
  rate: number;
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
const smooth = (edge0: number, edge1: number, x: number) => {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
};
/** Most rings per second, so fast songs in a high gear do not strobe. */
const MAX_RINGS_PER_SECOND = 4.5;

export function gearFor(
  energy: number,
  anticipation: number,
  rate: number,
  period: number,
): number {
  let gear = energy < 0.38 ? 0.5 : energy < 0.72 ? 1 : 2;
  if (anticipation > 0.45) gear = Math.min(2, gear * 2);
  if (rate < 0.85) gear /= 2;
  else if (rate > 1.2) gear *= 2;
  while (gear > 0.25 && gear / period > MAX_RINGS_PER_SECOND) gear /= 2;
  return Math.max(0.25, gear);
}

export class TunnelMotion {
  /** Smoothed position on the beat grid, in beats. */
  beat = 0;
  /** Seconds per beat, following the local tempo. */
  period = 0.5;
  travel = 0;
  gear = 1;
  rotation = 0;
  /** How much of each bar's turn lands on the downbeat, 0..1. */
  punch = 0.5;

  private offset = 0;
  private bar = Number.NEGATIVE_INFINITY;
  private measured: (BeatClock & { at: number }) | null = null;
  private started = false;
  private rotationBase = 0;
  private rotationFrom = 0;
  private direction = 1;
  private sector = Math.PI / 6;
  private section = -1;

  receive(clock: BeatClock | null, nowSeconds: number) {
    this.measured = clock ? { ...clock, at: nowSeconds } : null;
  }

  /** Bars turned so far as one continuous measure, eased toward each downbeat by punch. */
  private turned(beat: number) {
    const bars = beat / 4,
      whole = Math.floor(bars),
      part = bars - whole;
    return whole + part + (smooth(0, 0.2, part) - part) * this.punch;
  }

  step(dt: number, nowSeconds: number, input: TunnelStep) {
    dt = Math.max(0, Math.min(0.1, dt));
    if (!input.audible) return;
    const clock = this.measured && nowSeconds - this.measured.at < 0.5 ? this.measured : null;
    let jumped = !this.started;
    if (clock) {
      const target = clock.position + (nowSeconds - clock.at) / clock.period;
      this.period += (clock.period - this.period) * (1 - Math.exp(-dt * 4));
      if (jumped || Math.abs(target - this.beat) > 1) {
        this.beat = target;
        this.period = clock.period;
        jumped = true;
      } else {
        // Predict, then pull gently toward the measured position: playback time arrives in steps.
        const predicted = this.beat + dt / this.period;
        this.beat = Math.max(
          this.beat,
          predicted + (target - predicted) * (1 - Math.exp(-dt * 10)),
        );
      }
    } else {
      this.period = 60 / Math.max(50, Math.min(220, input.bpm || 100));
      this.beat += dt / this.period;
    }
    this.started = true;

    const bar = Math.floor(this.beat / 4);
    if (jumped || bar !== this.bar) {
      const gear = gearFor(input.energy, input.anticipation, input.rate, this.period);
      if (jumped || gear !== this.gear) {
        this.gear = gear;
        // A whole offset keeps rings crossing on the beat.
        this.offset = Math.round(this.travel - this.beat * this.gear);
      }
      this.bar = bar;
    }
    this.travel = this.offset + this.beat * this.gear;

    this.punch += (clamp01((input.energy - 0.3) / 0.5) - this.punch) * (1 - Math.exp(-dt * 0.8));
    if (input.section !== this.section) {
      this.rotationBase = this.rotation;
      this.rotationFrom = this.turned(this.beat);
      this.direction = input.section % 2 ? -1 : 1;
      this.sector = Math.PI / Math.max(3, Math.abs(input.sides) || 6);
      this.section = input.section;
    }
    this.rotation =
      this.rotationBase +
      this.direction * this.sector * (this.turned(this.beat) - this.rotationFrom);
  }

  reset() {
    this.beat = this.travel = this.rotation = this.offset = 0;
    this.period = 0.5;
    this.gear = 1;
    this.punch = 0.5;
    this.bar = Number.NEGATIVE_INFINITY;
    this.measured = null;
    this.started = false;
    this.rotationBase = this.rotationFrom = 0;
    this.direction = 1;
    this.section = -1;
  }
}
