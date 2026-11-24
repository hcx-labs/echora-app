export const CALIBRATION_SAMPLES = "/analysis/library/recording/calibration/samples";
export type RecordingAvailability = { enabled: boolean; recognition_enabled?: boolean; calibration_enabled?: boolean; indexed_tracks: number; reason?: string };
export function isCalibrationOnly(status: RecordingAvailability | null) {
  return status?.recognition_enabled === false && status.calibration_enabled === true;
}
export function calibrationRequest(audio: Blob, trackId: string | undefined, negative: boolean, notes: string) {
  if (!negative && !trackId) throw new Error("Choose the actual library track or mark Not in my library.");
  const query = new URLSearchParams(negative ? { not_in_library: "true" } : { expected_track_id: trackId! });
  if (notes.trim().length > 300) throw new Error("Keep calibration notes under 300 characters.");
  return { url: `${CALIBRATION_SAMPLES}?${query}`, init: { method: "POST", headers: { "Content-Type": audio.type || "application/octet-stream", "X-Echora-Calibration-Consent": "save-for-calibration-v1", "X-Echora-Calibration-Notes": encodeURIComponent(notes.trim()) }, body: audio } };
}
export function validateCalibrationAudio(size: number, duration: number) {
  if (!size || size > 8 * 1024 * 1024) throw new Error("Choose a usable excerpt no larger than 8 MiB.");
  if (!Number.isFinite(duration) || duration <= 0 || duration > 20) throw new Error("Choose an excerpt no longer than 20 seconds.");
}
export async function checkCalibrationAudio(audio: Blob) {
  validateCalibrationAudio(audio.size, 1);
  const context = new AudioContext();
  try {
    let decoded: AudioBuffer;
    try { decoded = await context.decodeAudioData(await audio.arrayBuffer()); }
    catch { throw new Error("This browser could not read the audio. Try a WAV, MP3, or another supported audio file."); }
    validateCalibrationAudio(audio.size, decoded.duration);
  } finally { await context.close(); }
}
