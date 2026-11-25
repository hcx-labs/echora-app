// Base path for browser-facing media (cover art, audio streams). The value is
// injected at runtime by the root layout from the ECHORA_MEDIA_BASE env var,
// so deployments can change the proxy shape without rebuilding the image.
//
// The path points at whatever routes to the analysis service *without* going
// through Node. Examples:
//   reverse proxy:  echora.com/api -> analysis:8000, ECHORA_MEDIA_BASE=/api
//   same host:      ECHORA_MEDIA_BASE=http://host:8000  (cookie + img work
//                   cross-origin without CORS; palette fetch needs CORS)
//   default:        /analysis (Next.js rewrite through Node, dev convenience)

type MediaBase = string & { __brand?: never };
declare global {
  interface Window { __ECHORA_MEDIA_BASE__?: MediaBase }
}

function base(): string {
  if (typeof window !== "undefined" && window.__ECHORA_MEDIA_BASE__) {
    return window.__ECHORA_MEDIA_BASE__.replace(/\/+$/, "");
  }
  return "/analysis";
}

export function mediaUrl(path: string) {
  const clean = path.startsWith("/analysis/") ? path.slice("/analysis".length) : path;
  return `${base()}${clean}`;
}
