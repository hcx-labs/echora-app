// Motion artwork for the full-screen player: a looping video of the album cover, when one has been
// generated. Kept free of imports so the logic can be tested directly.

export type ArtworkStyle = "still" | "motion";
export const artworkStyleStorageKey = "echora:artwork-style";

/** The saved choice wins; otherwise motion, unless the system asks for reduced motion. */
export function resolveArtworkStyle(stored: string | null, reducedMotion: boolean): ArtworkStyle {
  if (stored === "still" || stored === "motion") return stored;
  return reducedMotion ? "still" : "motion";
}

export function motionArtworkPath(trackId: string) {
  return `/analysis/library/tracks/${encodeURIComponent(trackId)}/motion-artwork`;
}

/** The video path from the lookup response, or null. Only Echora's own video route is accepted. */
export function motionVideoPath(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const { available, url } = body as { available?: unknown; url?: unknown };
  if (available !== true || typeof url !== "string") return null;
  return /^\/motion-artwork\/[0-9a-f-]{36}\.mp4$/.test(url) ? url : null;
}
