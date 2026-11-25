"use client";

import { useEffect, useState } from "react";

export type SidebarCuration = { id: string; name: string; tracks: unknown[] };
const changedEvent = "echora:curations-changed";

export function publishCurations(curations: SidebarCuration[]) {
  window.dispatchEvent(new CustomEvent(changedEvent, { detail: curations }));
}

export function useSidebarCurations(onboarding: boolean) {
  const [curations, setCurations] = useState<SidebarCuration[]>([]);
  useEffect(() => {
    if (onboarding) return;
    const abort = new AbortController();
    let updated = false;
    const changed = (event: Event) => {
      updated = true;
      setCurations((event as CustomEvent<SidebarCuration[]>).detail);
    };
    window.addEventListener(changedEvent, changed);
    fetch("/analysis/library/curations", { signal: abort.signal })
      .then(response => response.ok ? response.json() : null)
      .then(body => {
        if (body && !abort.signal.aborted && !updated) setCurations(body.curations || []);
      }).catch(() => {});
    return () => {
      abort.abort();
      window.removeEventListener(changedEvent, changed);
    };
  }, [onboarding]);
  return curations;
}
