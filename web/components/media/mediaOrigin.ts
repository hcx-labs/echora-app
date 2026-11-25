// Where browser-facing media (cover art, audio streams, motion artwork) is fetched from. The
// value is injected at runtime by the root layout from the ECHORA_MEDIA_BASE env var, so
// deployments can change the proxy shape without rebuilding the image.
//
// ECHORA_MEDIA_BASE is the origin that serves the backend's /api/v1 directly, without Node:
//   same origin (default, ""): nginx (or any reverse proxy) routes /api/v1/ to the backend
//   other origin:  ECHORA_MEDIA_BASE=http://host:8000  (cookie + img work cross-origin without
//                  CORS; palette fetch needs CORS)
// In `next dev` without a proxy, /api/v1 goes through the Next.js rewrite.

const API = "/api/v1";

type MediaBase = string & { __brand?: never };
declare global {
  interface Window {
    __ECHORA_MEDIA_BASE__?: MediaBase;
  }
}

function base(): string {
  if (typeof window !== "undefined" && window.__ECHORA_MEDIA_BASE__) {
    return window.__ECHORA_MEDIA_BASE__.replace(/\/+$/, "");
  }
  return "";
}

/** An API path ("/media/...", with or without the /api/v1 prefix) on the media origin. */
export function mediaUrl(path: string) {
  const clean = path.startsWith(`${API}/`) ? path.slice(API.length) : path;
  return `${base()}${API}${clean}`;
}
