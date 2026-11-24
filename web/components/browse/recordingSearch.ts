export const MAX_RECORDING_BYTES = 8 * 1024 * 1024;
export const MAX_RECORDING_SECONDS = 20;
export type RecordingQuality = "high" | "possible" | "low";
export function recordingQualityLabel(quality: RecordingQuality) {
  return quality === "high" ? "High-quality match" : quality === "low" ? "Low-quality match" : "Possible match";
}
export type RecordingMatch = { match_quality?: RecordingQuality; track_id: string; title: string; artist: string; album: string; score: number; offset_seconds: number; connection_id?: string; source_id?: string; cover_art?: string; duration_seconds?: number };
export type RecordingJob = { status: "queued" | "running" | "complete" | "failed" | "cancelled" | "partial"; result?: { state: "identified" | "ambiguous" | "no_match" | "insufficient_audio"; matches: RecordingMatch[] }; error?: string };
export function confirmedMatch(job: RecordingJob | null) {
  return job?.status === "complete" && job.result?.state === "identified" ? job.result.matches[0] : undefined;
}
export type RecordingTrack = { id: string; title: string; artist?: string; album?: string; duration_seconds: number; connection_id?: string; source_id?: string; cover_art?: string; matched_at_seconds: number; matched_source: "recording"; recording_score: number; recording_confirmed: boolean; recording_quality: RecordingQuality };
export function recordingTracks(job: RecordingJob): RecordingTrack[] {
  if (job.status !== "complete" || !job.result || !["identified", "ambiguous"].includes(job.result.state)) return [];
  return job.result.matches.map((match, index) => ({ id: match.track_id, title: match.title, artist: match.artist, album: match.album,
    connection_id: match.connection_id, duration_seconds: match.duration_seconds ?? 0, source_id: match.source_id, cover_art: match.cover_art,
    matched_at_seconds: match.offset_seconds, matched_source: "recording",
    recording_score: match.score, recording_confirmed: job.result?.state === "identified" && index === 0,
    recording_quality: match.match_quality ?? (job.result?.state === "identified" && index === 0 ? "high" : "possible") }));
}
export function recordingWarning(job: RecordingJob): string {
  if (job.status !== "complete" || job.result?.state !== "ambiguous" || !job.result.matches.length) return "";
  if (job.result.matches[0].match_quality === "low") return "Low-quality match evidence. Listen to check the candidates or try another excerpt. Scores are similarity measures, not confidence percentages.";
  return job.result.matches.length > 1
    ? "Possible matches, best score first. Identification is uncertain. Listen to the candidates or try a different excerpt. Scores are similarity measures, not confidence percentages."
    : "Possible match. The song or its position is uncertain; the evidence may be weak or ambiguous. Listen to check or try a different excerpt. The score is not a confidence percentage.";
}
export function curationHref(trackId: string, intent: "similar" | "journey") {
  return `/curate?${new URLSearchParams({ recording_track: trackId, recording_intent: intent })}`;
}
export function recordingMessage(job: RecordingJob) {
  if (job.status === "queued") return "Queued for recording identification…";
  if (job.status === "running") return "Identifying the recording…";
  if (job.status === "failed") return job.error || "Recording search failed. Try another recording.";
  if (job.status === "cancelled") return "Recording search was cancelled.";
  if (job.status === "partial") return "Search finished with partial results. No identification is confirmed.";
  switch (job.result?.state) {
    case "identified": return "Recording identified";
    case "ambiguous": return "Identification is uncertain. The song or its position may be ambiguous. Try a different excerpt.";
    case "no_match": return "No matching recording in your indexed library.";
    case "insufficient_audio": return "Not enough usable audio. Try a clearer or longer excerpt.";
    default: return "The search returned no usable result. Try again.";
  }
}

export function resultConnection(track: { matched_source?: string; connection_id?: string }, selected: string) {
  return track.matched_source === "recording" ? track.connection_id || "" : selected;
}
export function canCurateRecording(track: { recording_confirmed?: boolean; recording_quality?: RecordingQuality }, index: number) {
  return index === 0 && track.recording_confirmed === true && track.recording_quality === "high";
}
