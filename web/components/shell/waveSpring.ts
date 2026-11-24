export type WaveSpring = { position: number; velocity: number };

/** Damped motion integrated at a fixed maximum step, independent of render FPS. */
export function advanceWaveSpring(state: WaveSpring, target: number, seconds: number, frequency: number) {
  const duration = Math.min(.1, Math.max(0, seconds));
  const steps = Math.max(1, Math.ceil(duration * 240));
  const dt = duration / steps;
  const damping = .82;
  for (let i = 0; i < steps; i++) {
    const acceleration = frequency * frequency * (target - state.position) - 2 * damping * frequency * state.velocity;
    state.velocity += acceleration * dt;
    state.position += state.velocity * dt;
  }
  return state.position;
}
