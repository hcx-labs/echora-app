
import { runtimeConfig } from "../runtime/runtimeConfig";
export type Job = { id?: string; kind?: string; parent_id?: string; job_id: string; connection_id?: string; curation_id?: string; status: string; dismissed_at?: string | null; phase: string; completed: number; total: number; message?: string; error?: string; cancel_requested?: boolean; unit?: string; track?: { id: string; title: string; artist?: string }; summary?: Record<string, number>; progress?: Record<string, unknown> };
export const isActiveJob = (job: Job | null) => !!job && ["queued", "running", "waiting"].includes(job.status);
export const isTerminalJob = (job: Job | null) => !!job && ["complete", "partial", "failed", "cancelled"].includes(job.status);
export async function jobRequest<T>(path: string, signal: AbortSignal, method = "GET"): Promise<T> {
  const response = await fetch(`/analysis${path}`, { signal, method });
  const body = await response.json();
  if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`);
  return body as T;
}
export async function discoverJob(connectionId: string, signal: AbortSignal): Promise<Job | null> {
  const query = new URLSearchParams({ connection_id: connectionId, library_only: "true" });
  // Curation work must never be presented as library synchronization.
  for (const active of [true, false]) {
    const { jobs } = await jobRequest<{ jobs: Job[] }>(`/jobs?${query}&active_only=${active}`, signal);
    for (const job of jobs) {
      if (job.curation_id || job.kind === "curation_refresh" || job.parent_id) continue;
      // Acknowledgement also applies to cancelled results. Never uncover older history.
      if (job.dismissed_at && isTerminalJob(job)) return null;
      if (job.status !== "cancelled") return job;
    }
  }
  return null;
}

/** One in-flight request at a time; failures keep the same job and retry. */
export function watchJob(connectionId: string, jobId: string | null, receive: (job: Job | null) => void, onError: (message: string) => void, delay?: number) {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout>;
  let id = jobId;
  async function poll() {
    let again = true;
    try {
      const job = id ? await jobRequest<Job>(`/jobs/${encodeURIComponent(id)}`, controller.signal) : await discoverJob(connectionId, controller.signal);
      if (controller.signal.aborted) return;
      receive(job); onError("");
      id = job?.job_id || job?.id || null;
      again = isActiveJob(job);
    } catch (reason) {
      if (controller.signal.aborted) return;
      onError(`Connection error: ${reason instanceof Error ? reason.message : "Could not read job progress"}. Retrying…`);
    }
    if (again && !controller.signal.aborted) timer = setTimeout(poll, delay ?? runtimeConfig().job_poll_ms);
  }
  void poll();
  return () => { controller.abort(); clearTimeout(timer); };
}


/** Legacy parent `completed` counts included cancelled batches. Never label those successful. */
export function jobPresentation(job: Job | null) {
  const fusion = job?.kind === "semantic_fusion_build";
  const fused = fusion && job?.status === "complete" && typeof job.summary?.fused === "number" ? job.summary.fused : null;
  const total = fused !== null ? (job?.summary?.total ?? fused) : job?.total || 0;
  const batch = job?.unit === "batches";
  const counts = job?.summary || job?.progress || {};
  const successful = fused ?? (batch ? (typeof counts.complete === "number" ? counts.complete : 0) : (job?.completed || 0));
  const percent = total ? Math.min(100, Math.max(0, Math.round(successful / total * 100))) : 0;
  const showPercent = !!job && total > 0 && (isActiveJob(job) || job.status === "complete");
  let detail = `${successful} of ${total} ${batch ? "batches successful" : fusion ? "vectors" : job?.unit || "tracks"}`;
  let message = batch ? detail : job?.message || "Waiting for worker";
  if (isTerminalJob(job) && job?.status !== "cancelled") {
    const label = fusion ? "Semantic fusion" : "Library processing";
    message = job?.status === "complete" ? `${label} complete` : job?.status === "partial" ? `${label} finished with errors` : `${label} failed`;
    if (fused !== null) detail = fused === 0 ? "No paired audio and lyrics embeddings to fuse." : `${fused} fused vectors stored across the Echora library.`;
    else if (!batch) detail = job?.status === "complete" ? "Processing finished." : "Review the error and results before retrying.";
  }
  const summary: string[] = [];
  if (batch) {
    for (const [key, label] of [["complete", "successful"], ["partial", "partially successful"], ["failed", "failed"], ["cancelled", "cancelled"]]) {
      const count = counts[key];
      if (typeof count === "number" && (count > 0 || key === "complete")) summary.push(`${count} ${label}`);
    }
  } else {
    for (const [key, label] of [["inserted", "new"], ["already_linked", "reused"], ["failed", "failed"], ["waveforms_generated", "waveforms generated"], ["melody_indexed", "melodies indexed"], ["lyrics_embedded", "lyrics embedded"], ["karaoke_aligned", "karaoke aligned"], ["voice_classified", "vocals classified"], ["unlinked", "unlinked"]]) {
      const count = job?.summary?.[key];
      if (typeof count === "number") summary.push(`${count} ${label}`);
    }
  }
  const recordingCount = counts.recording_fingerprinted;
  if (typeof recordingCount === "number") summary.push(`${recordingCount} recording fingerprints generated`);
  return { percent, showPercent, detail, message, summary };
}
