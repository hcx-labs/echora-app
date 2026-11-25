"use client";

import { usePathname } from "next/navigation";

/**
 * URL helpers for views that live inside a page: a settings section, an open
 * curation, a selected galaxy track. Next keeps usePathname/useSearchParams in
 * sync with the native History API, so updating the URL here re-renders
 * readers without a server round trip, and Back/Forward just work.
 */

/** The single path segment after `/<section>/`, decoded, or null. */
export function useSubPath(section: string): string | null {
  const pathname = usePathname() || "";
  const prefix = `/${section}/`;
  if (!pathname.startsWith(prefix)) return null;
  const segment = pathname.slice(prefix.length).split("/")[0];
  return segment ? decodeURIComponent(segment) : null;
}

/**
 * Point the address bar at `url`. Push for a destination the reader may want
 * to go Back from; replace for refinements such as filters.
 */
export function setUrl(url: string, mode: "push" | "replace" = "push") {
  const current = `${window.location.pathname}${window.location.search}`;
  if (url === current) return;
  // Pass null, not history.state: Next's own entries carry an internal marker, and
  // reusing it makes Next treat this as its own call and skip updating usePathname.
  // Next copies its internal state across by itself.
  if (mode === "push") window.history.pushState(null, "", url);
  else window.history.replaceState(null, "", url);
}
