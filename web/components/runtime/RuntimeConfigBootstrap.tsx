"use client";

import { useEffect } from "react";
import { applyRuntimeConfig, runtimeConfig } from "./runtimeConfig";

export default function RuntimeConfigBootstrap() {
  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), runtimeConfig().recording_request_timeout_ms);
    void fetch("/analysis/settings/runtime", { cache: "no-store", signal: controller.signal })
      .then(async response => {
        if (!response.ok) return;
        const value: unknown = await response.json();
        if (!controller.signal.aborted) applyRuntimeConfig(value);
      })
      .catch(() => { /* Generated defaults keep the UI usable during an API outage. */ })
      .finally(() => window.clearTimeout(timeout));
    return () => { controller.abort(); window.clearTimeout(timeout); };
  }, []);
  return null;
}
