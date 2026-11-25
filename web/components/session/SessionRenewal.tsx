"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { SessionActivity, type Cadence, type PlaybackSample } from "./sessionActivity";
import { invalidateSessionUser, SESSION_EXPIRED_EVENT } from "./sessionUser";

const CADENCE_KEY = "echora:session-cadence";
function readCadence(): Cadence | null {
  try {
    const value = JSON.parse(localStorage.getItem(CADENCE_KEY) || "null");
    return value && typeof value.absoluteExpiresAt === "string" && Number.isFinite(value.dueAt) ? value : null;
  } catch { return null; }
}
function writeCadence(value: Cadence) {
  // Only scheduling metadata, never session tokens or identity information.
  try { localStorage.setItem(CADENCE_KEY, JSON.stringify(value)); } catch { /* Storage may be disabled. */ }
}

export default function SessionRenewal() {
  const pathname = usePathname();
  const router = useRouter();
  const enabled = pathname !== "/login" && pathname !== "/";
  useEffect(() => {
    if (!enabled) return;
    const tracker = new SessionActivity({
      now: Date.now,
      visible: () => document.visibilityState === "visible",
      request: (activity, signal) => fetch(`/analysis/auth/session${activity ? "/activity" : ""}`, {
        method: activity ? "POST" : "GET", credentials: "same-origin", cache: "no-store", signal,
        ...(activity ? { headers: { "X-Echora-Activity": "1" } } : {}),
      }),
      schedule: (callback, delay) => window.setTimeout(callback, delay),
      cancel: timer => window.clearTimeout(timer as number | undefined),
      readCadence, writeCadence,
      exclusive: async work => {
        if (navigator.locks) await navigator.locks.request("echora:session-activity", { ifAvailable: true }, async lock => { if (lock) await work(); });
        else await work();
      },
      expired: () => { invalidateSessionUser(); router.replace("/login"); },
    });
    const gesture = (event: Event) => tracker.gesture(event.isTrusted);
    const playback = (event: Event) => tracker.playback((event as CustomEvent<PlaybackSample>).detail);
    const expired = () => tracker.stop();
    const gestures = ["pointerdown", "keydown", "wheel", "touchstart"] as const;
    gestures.forEach(name => window.addEventListener(name, gesture, { passive: true, capture: true }));
    window.addEventListener("focus", gesture);
    document.addEventListener("visibilitychange", gesture);
    window.addEventListener("echora:session-playback", playback);
    window.addEventListener(SESSION_EXPIRED_EVENT, expired);
    tracker.start();
    return () => {
      tracker.stop();
      gestures.forEach(name => window.removeEventListener(name, gesture, true));
      window.removeEventListener("focus", gesture);
      document.removeEventListener("visibilitychange", gesture);
      window.removeEventListener("echora:session-playback", playback);
      window.removeEventListener(SESSION_EXPIRED_EVENT, expired);
    };
  }, [enabled, router]);
  return null;
}
