/**
 * A smooth playback clock for lyrics. Browsers update `audio.currentTime` in
 * coarse steps, so between updates time is extrapolated from the wall clock and
 * gently slewed back toward the reported position. It never runs backwards
 * except on a real jump (seek, stall or track change).
 */
export type ClockAnchor = { media: number; at: number; last: number };

/** Larger disagreements than this are treated as a jump, not jitter. */
const JUMP_SECONDS = .25;
/** Share of the remaining error corrected per frame: small enough to stay invisible. */
const SLEW = .1;

export function smoothPlaybackTime(anchor: ClockAnchor | null, media: number, now: number, running: boolean, rate = 1): { time: number; anchor: ClockAnchor | null } {
  if (!running) return { time: media, anchor: null };
  if (!anchor) return { time: media, anchor: { media, at: now, last: media } };
  const predicted = anchor.media + (now - anchor.at) / 1000 * rate;
  const error = media - predicted;
  if (Math.abs(error) > JUMP_SECONDS) return { time: media, anchor: { media, at: now, last: media } };
  const corrected = predicted + error * SLEW;
  const time = Math.max(corrected, anchor.last);
  return { time, anchor: { media: corrected, at: now, last: time } };
}

// A tiny external store so components can subscribe to the clock without
// putting a per-frame value into React state high up the tree.
let latest = 0;
const listeners = new Set<() => void>();

export function publishPlaybackTime(seconds: number) {
  if (seconds === latest) return;
  latest = seconds;
  listeners.forEach(listener => listener());
}
export function subscribePlaybackTime(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
export function playbackTimeSnapshot() { return latest; }
