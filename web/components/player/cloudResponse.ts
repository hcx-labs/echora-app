/** Input bands come from PlayerProvider: 20–250, 250–2000, 2000–10000 Hz. */
export type CloudAudio = {
  bass: number; mid: number; treble: number; onset?: boolean; bpm?: number | null;
  bassAttack?: number; midAttack?: number; trebleAttack?: number;
};
export type CloudSensitivity = { bassReactivity: number; vocalReactivity: number; trebleReactivity: number };
const clamp = (n: number) => Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0;

export class CloudResponse {
  bass = 0;
  mid = 0;
  treble = 0;
  glow = 0;
  strike = 0;
  travel = 0;
  seed = 1;
  bpm = 0;
  private speed = 0;
  private targets = [0, 0, 0];
  private lastAudio = -Infinity;
  private lastStrike = -Infinity;

  receive(audio: CloudAudio, sensitivity: CloudSensitivity, now: number) {
    this.lastAudio = now;
    this.targets = [
      clamp(audio.bass * sensitivity.bassReactivity * 1.65),
      clamp(audio.mid * sensitivity.vocalReactivity * 1.8),
      clamp(audio.treble * sensitivity.trebleReactivity * 2),
    ];
    this.bpm = audio.bpm ?? 0;
    const highAttack = clamp((audio.trebleAttack ?? 0) * sensitivity.trebleReactivity * 7);
    const midAttack = clamp((audio.midAttack ?? 0) * sensitivity.vocalReactivity * 6);
    const lowAttack = clamp((audio.bassAttack ?? 0) * sensitivity.bassReactivity * 6);
    this.glow = Math.max(this.glow, highAttack * .85 + midAttack * .35);
    const attack = Math.max(lowAttack, midAttack, highAttack);
    const kick = audio.onset && this.targets[0] > .18;
    // Any band's transient can strike. The cooldown limits repeated flashes,
    // while sustained loudness alone cannot keep retriggering lightning.
    if ((attack > .13 || kick) && now - this.lastStrike >= .8) {
      this.strike = Math.min(.8, .36 + attack * .55);
      this.seed += 7.13;
      this.lastStrike = now;
    }
  }

  step(dt: number, now: number, playing: boolean, rate: number) {
    dt = Math.max(0, Math.min(.1, dt));
    const audible = playing && now - this.lastAudio < .7;
    const blend = 1 - Math.exp(-dt * 2.5);
    this.bass += ((audible ? this.targets[0] : 0) - this.bass) * (1 - Math.exp(-dt * (this.targets[0] > this.bass ? 11 : 3)));
    this.mid += ((audible ? this.targets[1] : 0) - this.mid) * (1 - Math.exp(-dt * 5));
    this.treble += ((audible ? this.targets[2] : 0) - this.treble) * (1 - Math.exp(-dt * 8));
    this.glow *= Math.exp(-dt * 3.2);
    this.strike *= Math.exp(-dt * 7);
    if (!audible) { this.strike = 0; this.glow *= Math.exp(-dt * 5); }
    // Integrate drift instead of multiplying absolute time by a changing tempo.
    const targetSpeed = audible ? (.08 + this.bass * .36 + this.mid * .14 + this.treble * .08) * Math.max(.6, Math.min(2, this.bpm / 100)) : 0;
    this.speed += (targetSpeed - this.speed) * blend;
    this.travel += dt * this.speed * rate;
  }

  reset() {
    this.bass = this.mid = this.treble = this.glow = this.strike = this.speed = 0;
    this.targets = [0, 0, 0];
    this.lastAudio = this.lastStrike = -Infinity;
    this.bpm = 0;
  }
}
