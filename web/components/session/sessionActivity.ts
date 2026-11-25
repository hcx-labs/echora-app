// No credentials are read or stored here. The cookie remains HttpOnly.
export const BOOTSTRAP_SAFETY_TIMEOUT_MS = 10_000;
export type SessionInfo = {
  renewed: boolean;
  expires_at: string;
  absolute_expires_at: string;
  renew_after_seconds: number;
  activity_check_seconds: number;
  activity_window_seconds: number;
  request_timeout_seconds: number;
};
export type Cadence = { absoluteExpiresAt: string; dueAt: number };
export type PlaybackSample = { source: string; currentTime: number; paused: boolean; seeking: boolean; readyState: number };
type Dependencies = {
  now: () => number;
  visible: () => boolean;
  request: (activity: boolean, signal: AbortSignal) => Promise<{ status: number; ok: boolean; json: () => Promise<unknown> }>;
  schedule: (callback: () => void, delay: number) => unknown;
  cancel: (timer: unknown) => void;
  expired: () => void;
  readCadence: () => Cadence | null;
  writeCadence: (value: Cadence) => void;
  exclusive: (work: () => Promise<void>) => Promise<void>;
};

function sessionInfo(value: unknown): SessionInfo {
  if (!value || typeof value !== "object") throw new Error("Invalid session configuration");
  const info = value as SessionInfo;
  if (typeof info.renewed !== "boolean" || !Number.isFinite(Date.parse(info.expires_at)) || !Number.isFinite(Date.parse(info.absolute_expires_at)) ||
    !Number.isFinite(info.renew_after_seconds) || info.renew_after_seconds < 0 ||
    ![info.activity_check_seconds, info.activity_window_seconds, info.request_timeout_seconds].every(value => typeof value === "number" && Number.isFinite(value) && value > 0)) {
    throw new Error("Invalid session configuration");
  }
  return info;
}

export class SessionActivity {
  private deps: Dependencies;
  private info: SessionInfo | null = null;
  private stopped = false;
  private busy = false;
  private started = false;
  private dueAt = 0;
  private retryAt = 0;
  private gestureAt = -Infinity;
  private playbackAt = -Infinity;
  private sample: PlaybackSample | null = null;
  private requestController: AbortController | null = null;
  private timer: unknown;
  private timeout: unknown;

  constructor(deps: Dependencies) { this.deps = deps; }

  start() {
    if (this.started || this.stopped) return;
    this.started = true;
    void this.send(false);
  }

  gesture(trusted: boolean) {
    if (!trusted || !this.deps.visible() || this.stopped) return;
    this.gestureAt = this.deps.now();
    this.check();
  }

  playback(value: PlaybackSample) {
    const previous = this.sample;
    this.sample = value;
    // Native timeupdate samples, not the playing flag or requestAnimationFrame:
    // seeks, paused media, repeated times and track switches are not listening.
    if (!value.source || value.paused || value.seeking || value.readyState < 3 || !Number.isFinite(value.currentTime)) {
      this.sample = null;
      return;
    }
    if (previous && previous.source === value.source && value.currentTime > previous.currentTime) {
      this.playbackAt = this.deps.now();
      this.check();
    }
  }

  private recent() {
    const windowMs = (this.info?.activity_window_seconds ?? 0) * 1000;
    const now = this.deps.now();
    return (this.deps.visible() && now - this.gestureAt <= windowMs) || now - this.playbackAt <= windowMs;
  }

  check() {
    if (this.stopped || this.busy || this.deps.now() < this.retryAt || !this.recent()) return;
    if (!this.info) { void this.send(false); return; }
    if (this.deps.now() < this.nextDue()) return;
    this.busy = true;
    // Web Locks serialize tabs where supported. Recheck shared cadence inside
    // the lock; another tab's successful response may have moved the deadline.
    void this.deps.exclusive(async () => {
      if (!this.stopped && this.recent() && this.deps.now() >= this.nextDue()) await this.send(true);
    }).catch(() => {
      this.retryAt = this.deps.now() + this.info!.activity_check_seconds * 1000;
    }).finally(() => { this.busy = false; });
  }

  private nextDue() {
    const shared = this.deps.readCadence();
    return Math.max(this.dueAt, shared && shared.absoluteExpiresAt === this.info?.absolute_expires_at ? shared.dueAt : 0);
  }

  private publish(dueAt: number) {
    this.dueAt = dueAt;
    this.deps.writeCadence({ absoluteExpiresAt: this.info!.absolute_expires_at, dueAt });
  }

  private arm() {
    this.deps.cancel(this.timer);
    if (this.stopped || !this.info) return;
    this.timer = this.deps.schedule(() => { this.check(); this.arm(); }, this.info.activity_check_seconds * 1000);
  }

  private async send(activity: boolean) {
    if (this.stopped) return;
    this.busy = true;
    const controller = new AbortController();
    this.requestController = controller;
    const timeoutMs = this.info ? this.info.request_timeout_seconds * 1000 : BOOTSTRAP_SAFETY_TIMEOUT_MS;
    const retryMs = this.info ? this.info.activity_check_seconds * 1000 : BOOTSTRAP_SAFETY_TIMEOUT_MS;
    this.retryAt = this.deps.now() + retryMs;
    // Also throttle other tabs after network errors, not just successful renewals.
    if (activity) this.publish(Math.max(this.nextDue(), this.retryAt));
    this.timeout = this.deps.schedule(() => controller.abort(), timeoutMs);
    try {
      const response = await this.deps.request(activity, controller.signal);
      if (this.stopped || controller.signal.aborted) return;
      if (response.status === 401) { this.stop(); this.deps.expired(); return; }
      if (!response.ok) throw new Error("Session request failed");
      const info = sessionInfo(await response.json());
      if (this.stopped || controller.signal.aborted) return;
      this.info = info;
      const dueAt = this.deps.now() + info.renew_after_seconds * 1000;
      this.publish(Math.max(this.nextDue(), dueAt));
      this.retryAt = this.deps.now() + info.activity_check_seconds * 1000;
    } catch {
      // Keep auth intact on transient failures. Retry only with recent activity,
      // never faster than the server's check cadence (including after timeout).
      this.retryAt = this.deps.now() + retryMs;
      if (activity && !this.stopped) this.publish(Math.max(this.nextDue(), this.retryAt));
    } finally {
      this.deps.cancel(this.timeout);
      this.requestController = null;
      this.busy = false;
      this.arm();
    }
  }

  stop() {
    this.stopped = true;
    this.requestController?.abort();
    this.deps.cancel(this.timeout);
    this.deps.cancel(this.timer);
  }
}
