// Karaoke stems: one four-channel Opus stream per song (vocals left/right, then instrumental
// left/right), remixed in the browser. One stream keeps the two stems sample-locked, which two
// separately buffered <audio> elements cannot guarantee after seeks or stalls.

export const STEMS_TYPE = 'audio/webm; codecs="opus"';

/** Balance -1 (vocals only) … 0 (the song as released) … 1 (instrumental only). */
export function stemGains(balance: number): { vocals: number; instrumental: number } {
  const value = Number.isFinite(balance) ? Math.min(1, Math.max(-1, balance)) : 0;
  return {
    vocals: value > 0 ? 1 - value : 1,
    instrumental: value < 0 ? 1 + value : 1,
  };
}

/** Whether this browser can play the stems file and remix it with Web Audio. */
export function stemsSupported(): boolean {
  if (typeof window === "undefined" || typeof window.AudioContext === "undefined") return false;
  try {
    return document.createElement("audio").canPlayType(STEMS_TYPE) !== "";
  } catch {
    return false;
  }
}

// Short enough to follow a dragged slider, long enough not to click.
const GAIN_SMOOTHING_SECONDS = 0.03;

/**
 * Routes the player's <audio> element through Web Audio, built on first use.
 *
 * Once an element feeds a MediaElementAudioSourceNode it plays only through that graph, so the
 * mixer stays connected for the rest of the page: the original song goes straight to the output,
 * the stems through a four-way channel split and two gains.
 */
export class StemMixer {
  private context: AudioContext | null = null;
  private source: MediaElementAudioSourceNode | null = null;
  private splitter: ChannelSplitterNode | null = null;
  private vocals: GainNode | null = null;
  private instrumental: GainNode | null = null;
  private stems = false;
  private balance = 0;
  private readonly element: HTMLMediaElement;

  constructor(element: HTMLMediaElement) {
    this.element = element;
  }

  /** Whether the element already plays through Web Audio. */
  get active() {
    return this.source !== null;
  }

  private build() {
    if (this.context) return this.context;
    // Web Audio hears a cross-origin media origin (ECHORA_MEDIA_BASE) only through CORS. The
    // caller sets the next source right after routing, which loads with this mode.
    this.element.crossOrigin = "use-credentials";
    const context = new AudioContext({ latencyHint: "playback" });
    const source = context.createMediaElementSource(this.element);
    const splitter = context.createChannelSplitter(4);
    const pair = (left: number, right: number) => {
      const merger = context.createChannelMerger(2);
      splitter.connect(merger, left, 0);
      splitter.connect(merger, right, 1);
      const gain = context.createGain();
      merger.connect(gain).connect(context.destination);
      return gain;
    };
    this.vocals = pair(0, 1);
    this.instrumental = pair(2, 3);
    this.context = context;
    this.source = source;
    this.splitter = splitter;
    this.applyGains(true);
    this.connect();
    return context;
  }

  private connect() {
    if (!this.source || !this.context || !this.splitter) return;
    this.source.disconnect();
    // A stereo (or mono) original goes straight out: through the split it would lose channels.
    this.source.connect(this.stems ? this.splitter : this.context.destination);
  }

  private applyGains(immediately = false) {
    if (!this.context || !this.vocals || !this.instrumental) return;
    const gains = stemGains(this.balance);
    const now = this.context.currentTime;
    for (const [node, value] of [
      [this.vocals, gains.vocals],
      [this.instrumental, gains.instrumental],
    ] as const) {
      node.gain.cancelScheduledValues(now);
      if (immediately) node.gain.setValueAtTime(value, now);
      else node.gain.setTargetAtTime(value, now, GAIN_SMOOTHING_SECONDS);
    }
  }

  /** Play the element's next source as stems (true) or as the original song (false). */
  route(stems: boolean) {
    this.stems = stems;
    if (stems) this.build();
    this.connect();
    this.resume();
  }

  setBalance(balance: number) {
    this.balance = balance;
    this.applyGains();
  }

  /** Browsers start or suspend contexts outside user gestures; call from play and gestures. */
  resume() {
    if (this.context && this.context.state !== "running") this.context.resume().catch(() => {});
  }
}
