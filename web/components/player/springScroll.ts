/**
 * Critically damped spring for scroll positions: it eases in and out, never
 * overshoots, and can be retargeted mid-flight without a jerk because velocity
 * carries over.
 */
export type SpringState = { position: number; velocity: number };

/** Angular frequency: a long jump settles in about 0.85 s, a single line much sooner. */
const OMEGA = 10;

export function springStep(state: SpringState, target: number, dtSeconds: number): SpringState {
  // Sub-step so a dropped frame cannot destabilise the integration.
  let { position, velocity } = state;
  const steps = Math.max(1, Math.ceil(dtSeconds / (1 / 120)));
  const dt = dtSeconds / steps;
  for (let i = 0; i < steps; i++) {
    const acceleration = OMEGA * OMEGA * (target - position) - 2 * OMEGA * velocity;
    velocity += acceleration * dt;
    position += velocity * dt;
  }
  return { position, velocity };
}

export function springSettled(state: SpringState, target: number) {
  return Math.abs(target - state.position) < .5 && Math.abs(state.velocity) < 4;
}
