"use client";

import { runtimeConfig } from "../runtime/runtimeConfig";

import { LoaderCircle, Speech, Square, X } from "lucide-react";
import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import AudioVisualizer from "./AudioVisualizer";
import { HumAttempt } from "./humAttempt";
import { Button } from "../ui/button";
import styles from "./HumSearchButton.module.css";
import activityStyles from "./SearchActivity.module.css";

type Track = { id: string; title: string; artist?: string; album?: string; duration_seconds: number; source_id?: string; cover_art?: string; similarity?: number; matched_at_seconds?: number };
type IndexStatus = { status: "missing" | "building" | "complete" | "failed"; indexed_tracks: number; error?: string };

export default function HumSearchButton({ onResults, onError, onStart, ref }: { ref?: Ref<{ cancel: () => void }>; onStart?: () => void; onResults: (tracks: Track[]) => void; onError: (message: string) => void }) {
  const [index, setIndex] = useState<IndexStatus>({ status: "missing", indexed_tracks: 0 });
  const [state, setState] = useState<"idle" | "permission" | "recording" | "searching" | "building">("idle");
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null);
  const attemptRef = useRef<HumAttempt | null>(null);

  function cancel() {
    attemptRef.current?.cancel();
    attemptRef.current = null;
    setActiveStream(null);
    setState("idle");
  }
  useImperativeHandle(ref, () => ({ cancel }));

  const pollingIndex = state === "building" || index.status === "building";
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const response = await fetch("/analysis/library/hum/index", { signal: controller.signal });
        const value: IndexStatus | null = response.ok ? await response.json() : null;
        if (controller.signal.aborted) return;
        if (value) {
          setIndex(value);
          if (value.status !== "building") setState(current => current === "building" ? "idle" : current);
        }
      } catch { /* Index polling is independent of the cancelled search. */ }
      if (!controller.signal.aborted && pollingIndex) timer = setTimeout(refresh, runtimeConfig().hum_index_poll_ms);
    }
    if (pollingIndex) timer = setTimeout(refresh, runtimeConfig().hum_index_poll_ms);
    else void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [pollingIndex]);
  useEffect(() => () => attemptRef.current?.cancel(), []);

  async function start() {
    onStart?.();
    attemptRef.current?.cancel();
    const attempt = new HumAttempt();
    attemptRef.current = attempt;
    onError("");
    if (index.status !== "complete") {
      setState("building");
      try {
        const response = await fetch("/analysis/library/hum/index?track_limit=50", { method: "POST", signal: attempt.signal });
        if (!attempt.current) return;
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          if (!attempt.current) return;
          throw new Error(body.detail || "Could not build the hum index");
        }
        setIndex(current => ({ ...current, status: "building" }));
      } catch (error) {
        if (attempt.current) { setState("idle"); onError(error instanceof Error ? error.message : "Could not build the hum index"); }
      }
      return;
    }
    setState("permission");
    try {
      const input = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!attempt.attachStream(input)) return;
      setActiveStream(input);
      const chunks: Blob[] = [];
      const active = new MediaRecorder(input);
      attempt.recorder = active;
      active.ondataavailable = event => { if (attempt.current && event.data.size) chunks.push(event.data); };
      active.onerror = () => {
        if (!attempt.current) return;
        cancel(); onError("Audio recording failed. Try again.");
      };
      active.onstop = async () => {
        if (!attempt.current) return;
        attempt.release(); setActiveStream(null); setState("searching");
        try {
          const recording = new Blob(chunks, { type: active.mimeType });
          const response = await fetch("/analysis/library/hum/search?limit=10", { method: "POST", headers: { "content-type": active.mimeType || "application/octet-stream" }, body: recording, signal: attempt.signal });
          if (!attempt.current) return;
          const contentType = response.headers.get("content-type") || "";
          const body = contentType.includes("application/json") ? await response.json() : null;
          if (!attempt.current) return;
          if (!response.ok) throw new Error(body?.detail || `Hum search failed (${response.status})`);
          onResults(body?.tracks || []);
        } catch (error) {
          if (attempt.current) onError(error instanceof Error ? error.message : "Hum search failed");
        } finally { if (attempt.current) setState("idle"); }
      };
      active.start(); setState("recording");
    } catch {
      if (!attempt.current) return;
      attempt.release(); setActiveStream(null); setState("idle");
      onError("Microphone access is required for hum search");
    }
  }

  function stop() {
    const active = attemptRef.current?.recorder;
    if (active?.state === "recording") active.stop();
  }

  const building = state === "building" || index.status === "building";
  const cancellable = state === "permission" || state === "searching";
  const label = state === "recording" ? "Stop humming" : cancellable ? "Cancel hum search" : building ? `Indexing ${index.indexed_tracks || 0}/50` : index.status === "complete" ? "Hum to search" : "Build hum index";
  return <Button type="button" variant="outline" className={`${styles.button} ${state === "recording" ? styles.recording : ""} ${cancellable || building ? activityStyles.active : ""}`} onClick={cancellable ? cancel : state === "recording" ? stop : start} disabled={building && !cancellable} aria-label={label} title={label}>
    {state === "recording" && activeStream ? <AudioVisualizer stream={activeStream} /> : null}
    {state === "recording" ? <Square /> : cancellable ? <X aria-hidden="true" /> : building ? <LoaderCircle className={styles.spin} /> : <Speech aria-hidden="true" />}
    <span>{label}</span>
  </Button>;
}
