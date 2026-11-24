import type { Job } from "./durableJobs";

export type Batch = Job & { batch_number: number; track_count: number };
const stages: Record<string, string> = {
  queued: "Waiting for worker", starting: "Starting", planning: "Checking missing work",
  models: "Loading model", waveform: "Waveforms", muq: "MuQ semantics", mert: "MERT acoustics",
  melody: "Melody extraction", fingerprint: "Fingerprints", descriptors: "Audio descriptors",
  recording_fingerprint: "Recording fingerprints", recording_decode: "Preparing recording",
  recording_encode: "Analyzing recording", recording_match: "Identifying recording",
  lyrics: "Lyrics retrieval and embeddings", karaoke: "Karaoke alignment", voice: "Voice classification",
  "audio-profiles": "Audio profiles", complete: "Complete", partial: "Partial result", failed: "Failed",
  cancelled: "Cancelled",
};
export function batchStatus(batch: Batch) {
  const active = batch.status === "running";
  const stage = active ? stages[batch.phase] || batch.phase || "Starting" : stages[batch.status] || batch.status;
  const progress = active && batch.total > 0
    ? `${batch.completed} / ${batch.total} ${batch.unit || "tracks"} in this stage`
    : `${batch.track_count} songs in batch`;
  return { stage, progress, message: active ? batch.message : batch.error };
}
