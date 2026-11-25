type Reference = { id: string; title: string; artist?: string; album?: string };
export function parseRecordingHandoff(search: string) {
  const params = new URLSearchParams(search);
  const trackId = params.get("recording_track");
  const intent = params.get("recording_intent");
  if (!trackId || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(trackId) || (intent !== "similar" && intent !== "journey")) return null;
  return { trackId: trackId.toLowerCase(), intent };
}

// Resolve one authorized track; never trust title/artist metadata in the URL.
export async function resolveRecordingReference(trackId: string, signal: AbortSignal): Promise<Reference> {
  const params = new URLSearchParams({ limit: "1", track_id: trackId });
  const response = await fetch(`/analysis/library/tracks?${params}`, { signal });
  if (!response.ok) throw new Error("Could not load the recording from your library");
  const body: { tracks: Reference[] } = await response.json();
  const match = body.tracks.find(track => track.id.toLowerCase() === trackId);
  if (match) return { id: match.id, title: match.title, artist: match.artist, album: match.album };
  throw new Error("This recording is no longer available in your library");
}
