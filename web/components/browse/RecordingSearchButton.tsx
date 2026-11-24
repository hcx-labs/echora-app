"use client";

import { runtimeConfig } from "../runtime/runtimeConfig";

import { AudioLines, Square, X } from "lucide-react";
import { useEffect, useEffectEvent, useImperativeHandle, useRef, useState, type Ref } from "react";
import { RecordingAttempt } from "./recordingAttempt";
import AudioVisualizer from "./AudioVisualizer";
import { MAX_RECORDING_BYTES, MAX_RECORDING_SECONDS, recordingMessage, recordingTracks, recordingWarning, type RecordingJob, type RecordingTrack } from "./recordingSearch";
import { Button } from "../ui/button";
import styles from "./RecordingSearchButton.module.css";
import activityStyles from "./SearchActivity.module.css";

type Availability = { recognition_enabled: boolean; indexed_tracks: number; reason?: string };

export default function RecordingSearchButton({ onResults, onError, onStart, ref }: {
  ref?: Ref<{ cancel: () => void }>;
  onStart?: () => void;
  onResults: (tracks: RecordingTrack[], warning: string) => void;
  onError: (message: string) => void;
}) {
  const [availability, setAvailability] = useState<Availability | null>(null);
  const [phase, setPhase] = useState<"idle" | "permission" | "recording" | "uploading" | "searching">("idle");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [jobId, setJobId] = useState("");
  const [message, setMessage] = useState("");
  const recorder = useRef<MediaRecorder | null>(null);
  const media = useRef<MediaStream | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lifecycle = useRef(new RecordingAttempt(cancelJob));
  function cancelJob(id: string) {
    if (id) void fetch(`/analysis/library/recording/search/${encodeURIComponent(id)}`, { method: "DELETE", keepalive: true, signal: AbortSignal.timeout(runtimeConfig().recording_request_timeout_ms) }).catch(() => {});
  }
  useImperativeHandle(ref, () => ({ cancel: discard }));
  const callbacks = useRef({ onResults, onError });
  useEffect(() => { callbacks.current = { onResults, onError }; }, [onResults, onError]);

  function release() {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    media.current?.getTracks().forEach(track => track.stop());
    media.current = null;
  }

  useEffect(() => {
    const controller = new AbortController();
    fetch("/analysis/library/recording/status", { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(runtimeConfig().recording_request_timeout_ms)]), cache: "no-store" }).then(async response => {
      if (!response.ok) throw new Error("Recording search is unavailable");
      const value: Availability = await response.json();
      if (!controller.signal.aborted) setAvailability(value);
    }).catch(() => {
      if (!controller.signal.aborted) setAvailability({ recognition_enabled: false, indexed_tracks: 0, reason: "Recording search is unavailable" });
    });
    return () => {
      controller.abort();
      // Invalidate pending microphone permission and upload callbacks on unmount.
      // eslint-disable-next-line react-hooks/exhaustive-deps
      lifecycle.current.invalidate();
      if (recorder.current) { recorder.current.onstop = null; if (recorder.current.state !== "inactive") recorder.current.stop(); }
      release();
    };
  }, []);

  useEffect(() => {
    if (phase !== "recording") return;
    const interval = setInterval(() => setSeconds(value => Math.min(value + 1, MAX_RECORDING_SECONDS)), 1000);
    return () => clearInterval(interval);
  }, [phase]);

  const discardFromEffect = useEffectEvent(() => discard());

  useEffect(() => {
    if (!jobId || phase !== "searching") return;
    const controller = new AbortController();
    let next: ReturnType<typeof setTimeout>;
    const attempt = lifecycle.current.generation;
    let failures = 0;
    const deadline = setTimeout(() => { if (lifecycle.current.generation === attempt) { discardFromEffect(); callbacks.current.onError("Recording search timed out. Try again."); } }, runtimeConfig().recording_search_timeout_ms);
    async function poll() {
      try {
        const response = await fetch(`/analysis/library/recording/search/${encodeURIComponent(jobId)}`, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(runtimeConfig().recording_request_timeout_ms)]), cache: "no-store" });
        if (lifecycle.current.generation !== attempt || controller.signal.aborted) return;
        if ([401, 403, 404].includes(response.status)) {
          if (!controller.signal.aborted) { callbacks.current.onError("This search is no longer available. Sign in again or start another search."); discardFromEffect(); }
          return;
        }
        if (!response.ok) throw new Error("Could not check recording search. Retrying...");
        const value: RecordingJob = await response.json();
        if (controller.signal.aborted || lifecycle.current.generation !== attempt) return;
        failures = 0;
        setMessage(recordingMessage(value));
        callbacks.current.onError("");
        if (!["queued", "running"].includes(value.status)) {
          const tracks = recordingTracks(value);
          lifecycle.current.jobId = "";
          if (tracks.length) callbacks.current.onResults(tracks, recordingWarning(value));
          if (!tracks.length) callbacks.current.onError(recordingMessage(value));
          setPhase("idle");
          return;
        }
      } catch (reason) {
        if (controller.signal.aborted || lifecycle.current.generation !== attempt) return;
        if (++failures >= 3) { discardFromEffect(); callbacks.current.onError("Could not check recording search. Try again."); return; }
        callbacks.current.onError(reason instanceof Error ? reason.message : "Could not check recording search. Retrying...");
      }
      next = setTimeout(poll, runtimeConfig().recording_poll_ms);
    }
    void poll();
    return () => { controller.abort(); clearTimeout(next); clearTimeout(deadline); };
  }, [jobId, phase]);

  function discard() {
    lifecycle.current.invalidate();
    setJobId(""); setMessage("");
    if (recorder.current) { recorder.current.onstop = null; if (recorder.current.state !== "inactive") recorder.current.stop(); }
    recorder.current = null; release(); setStream(null); setPhase("idle");
  }

  async function submitAudio(audio: Blob, attempt: number) {
    if (!audio.size || audio.size > MAX_RECORDING_BYTES) {
      setPhase("idle"); callbacks.current.onError("Record a usable excerpt under 8 MiB."); return;
    }
    setPhase("uploading");
    try {
      const response = await fetch("/analysis/library/recording/search", { method: "POST", headers: { "content-type": audio.type || "application/octet-stream" }, body: audio, signal: AbortSignal.timeout(runtimeConfig().recording_upload_timeout_ms), keepalive: false });
      const body = await response.json().catch(() => null);
      if (response.status !== 202 || typeof body?.job_id !== "string" || !body.job_id) throw new Error(typeof body?.detail === "string" ? body.detail : `Could not start recording search (${response.status})`);
      if (!lifecycle.current.admit(attempt, body.job_id)) return;
      setJobId(body.job_id); setMessage("Queued for recording identification..."); setPhase("searching");
    } catch (reason) {
      if (lifecycle.current.generation !== attempt) return;
      callbacks.current.onError(reason instanceof Error ? reason.message : "Recording search failed"); setPhase("idle");
    }
  }

  async function start() {
    if (availability?.recognition_enabled !== true || phase !== "idle") return;
    onStart?.();
    const attempt = ++lifecycle.current.generation;
    callbacks.current.onError(""); setJobId(""); setSeconds(0); setPhase("permission");
    try {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") throw new Error("This browser cannot record audio. Use a supported browser over HTTPS.");
      const input = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      if (lifecycle.current.generation !== attempt) { input.getTracks().forEach(track => track.stop()); return; }
      media.current = input;
      const active = new MediaRecorder(input);
      recorder.current = active;
      const chunks: Blob[] = [];
      let bytes = 0;
      active.ondataavailable = event => {
        if (lifecycle.current.generation !== attempt) return;
        bytes += event.data.size;
        if (bytes > MAX_RECORDING_BYTES) { discard(); callbacks.current.onError("Recording exceeds 8 MiB. Try a shorter excerpt."); return; }
        if (event.data.size) chunks.push(event.data);
      };
      active.onerror = () => { if (lifecycle.current.generation === attempt) { discard(); callbacks.current.onError("Audio recording failed. Check your microphone and try again."); } };
      active.onstop = async () => {
        if (lifecycle.current.generation !== attempt) return;
        release(); recorder.current = null; setStream(null);
        await submitAudio(new Blob(chunks, { type: active.mimeType }), attempt);
      };
      active.start(250); setStream(input); setPhase("recording");
      timer.current = setTimeout(() => { if (active.state === "recording") active.stop(); }, MAX_RECORDING_SECONDS * 1000 - 250);
    } catch (reason) {
      if (lifecycle.current.generation !== attempt) return;
      release(); setStream(null); setPhase("idle");
      callbacks.current.onError(reason instanceof Error ? reason.message : "Microphone access is required for recording search");
    }
  }

  const busy = phase === "permission" || phase === "uploading" || phase === "searching";
  const label = !availability ? "Checking recording search" : availability.recognition_enabled !== true
    ? availability.reason || "Recording search is unavailable"
    : phase === "recording" ? `Stop recording (${seconds}s)`
    : phase === "permission" ? "Waiting for microphone permission"
    : phase === "uploading" ? "Uploading recording"
    : phase === "searching" ? message : "Record a playing song to search";
  return <><span role="status" className={styles.srOnly}>{phase !== "idle" ? label : message}</span><Button type="button" variant="outline" className={`${styles.trigger} ${phase === "recording" ? styles.recording : ""} ${busy ? activityStyles.active : ""}`}
    disabled={availability?.recognition_enabled !== true} aria-label={busy ? `Cancel: ${label}` : label} title={busy ? `Cancel: ${label}` : label}
    onClick={busy ? discard : phase === "recording" ? () => { if (recorder.current?.state === "recording") recorder.current.stop(); } : () => void start()}>
    {phase === "recording" && stream && <AudioVisualizer stream={stream} />}
    {phase === "recording" ? <Square aria-hidden="true" /> : busy ? <X aria-hidden="true" /> : <AudioLines aria-hidden="true" />}
    <span>{phase === "recording" ? `Stop (${seconds}s)` : busy ? "Cancel" : "Identify song"}</span>
  </Button></>;
}
