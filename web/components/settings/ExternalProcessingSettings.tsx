"use client";

import { toast } from "sonner";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Notice } from "../ui/notice";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { LoadingState } from "../ui/spinner";
import { SectionHeading, Toggle, Choice, ChoiceItem } from "./Presentation";
import { SettingRow } from "./SettingRow";
import { SettingsNotice } from "./SettingsNotice";
import styles from "./ExternalAISettings.module.css";
import layout from "./SettingsView.module.css";

type Setup = {
  id: string;
  status: string;
  error: string | null;
  phase?: string | null;
  message?: string | null;
  completed?: number | null;
  total?: number | null;
};
type Configuration = {
  enabled: boolean;
  token_id: string;
  has_secret: boolean;
  has_hf_token: boolean;
  gpu: string;
  default_compute: "local" | "modal";
  allow_users: boolean;
  workspace: string | null;
  status: "unprepared" | "preparing" | "ready" | "failed";
  status_detail: string | null;
  checked_at: string | null;
  image: string;
  gpu_types: string[];
  setup: Setup | null;
};

const endpoint = "/analysis/settings/external-processing";
// The fields an admin edits on this page; everything else is status from the server.
const editable = ["enabled", "token_id", "gpu", "default_compute", "allow_users"] as const;
type Editable = Pick<Configuration, (typeof editable)[number]>;
const editsOf = (value: Configuration): Editable =>
  Object.fromEntries(editable.map((key) => [key, value[key]])) as Editable;
const sameEdits = (a: Configuration, b: Configuration) =>
  editable.every((key) => a[key] === b[key]);
const statusText: Record<Configuration["status"], string> = {
  unprepared: "Not prepared",
  preparing: "Preparing",
  ready: "Ready",
  failed: "Preparation failed",
};
// Approximate on-demand prices, for orientation only; Modal's pricing page is authoritative.
const gpuHints: Record<string, string> = {
  T4: "16 GB, slowest",
  L4: "24 GB",
  A10G: "24 GB",
  L40S: "48 GB, recommended",
  "A100-40GB": "40 GB",
  "A100-80GB": "80 GB",
  H100: "80 GB, fastest",
};

async function request<T>(path = "", options?: RequestInit): Promise<T> {
  const response = await fetch(endpoint + path, {
    ...options,
    cache: "no-store",
    credentials: "same-origin",
  });
  // Never display response bodies verbatim: validation payloads must not expose the token.
  if (!response.ok) {
    if (response.status === 401) throw new Error("Your session expired. Sign in again.");
    if (response.status === 403) throw new Error("Administrator access is required.");
    if (response.status === 409) throw new Error("Enable Modal and save a token first.");
    if (response.status === 422) {
      const body = await response.json().catch(() => null);
      const detail = typeof body?.detail === "string" ? body.detail : "";
      throw new Error(detail || "Check the token ID, secret and GPU type.");
    }
    throw new Error("Could not reach External processing settings. Try again.");
  }
  return response.json();
}

function formatTime(value: string | null) {
  return value ? new Date(value).toLocaleString() : "never";
}

export default function ExternalProcessingSettings() {
  const [value, setValue] = useState<Configuration | null>(null);
  // The configuration as last saved; Prepare acts on this, not on unsaved edits.
  const [saved, setSaved] = useState<Configuration | null>(null);
  const savedRef = useRef<Configuration | null>(null);
  const [secret, setSecret] = useState("");
  const [clearSecret, setClearSecret] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [starting, setStarting] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [loadError, setLoadError] = useState("");

  const refresh = useCallback(
    (signal?: AbortSignal) =>
      request<Configuration>("", { signal }).then((next) => {
        if (signal?.aborted) return next;
        // Refreshes (also every 2 s while preparing) update status but keep unsaved edits.
        const previous = savedRef.current;
        savedRef.current = next;
        setSaved(next);
        setValue((current) =>
          current && previous && !sameEdits(current, previous)
            ? { ...next, ...editsOf(current) }
            : next,
        );
        return next;
      }),
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal)
      .catch((reason) => {
        if (!controller.signal.aborted)
          setLoadError(reason instanceof Error ? reason.message : "Could not load settings.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [attempt, refresh]);

  const setupActive = Boolean(
    value?.setup && ["queued", "running", "waiting"].includes(value.setup.status),
  );
  // Follow a running preparation until it finishes.
  useEffect(() => {
    if (!setupActive) return;
    const timer = window.setInterval(() => {
      refresh().catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [setupActive, refresh]);

  function update(patch: Partial<Configuration>) {
    setValue((current) => (current ? { ...current, ...patch } : current));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!value || saving) return;
    if (value.token_id && !value.token_id.trim().startsWith("ak-")) {
      toast.error("Check these settings", { description: "Modal token IDs start with ak-." });
      return;
    }
    setSaving(true);
    try {
      const next = await request<Configuration>("", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          enabled: value.enabled,
          token_id: value.token_id.trim(),
          gpu: value.gpu,
          default_compute: value.default_compute,
          allow_users: value.allow_users,
          ...(clearSecret ? { token_secret: "" } : secret ? { token_secret: secret } : {}),
        }),
      });
      setValue(next);
      savedRef.current = next;
      setSaved(next);
      setSecret("");
      setClearSecret(false);
      toast.success("External processing saved", {
        description:
          next.enabled && next.status !== "ready"
            ? "Prepare Modal now, or let the first Modal sync prepare it."
            : undefined,
      });
    } catch (reason) {
      toast.error("Could not save External processing", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  async function prepare() {
    if (starting) return;
    setStarting(true);
    try {
      await request("/prepare", { method: "POST" });
      await refresh();
      toast.success("Preparing Modal", {
        description: "Echora is deploying to your workspace and downloading models.",
      });
    } catch (reason) {
      toast.error("Could not start preparation", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setStarting(false);
    }
  }

  const setup = value?.setup;
  const unsaved = Boolean(value && saved && (!sameEdits(value, saved) || secret || clearSecret));
  const progress =
    setup && setupActive && setup.total
      ? Math.round(((setup.completed ?? 0) / setup.total) * 100)
      : null;

  return (
    <form className={styles.form} onSubmit={save} aria-busy={loading || saving}>
      {loading ? (
        <LoadingState label="Loading External processing settings…" />
      ) : (
        value && (
          <div className={styles.sections}>
            <fieldset disabled={saving}>
              <section aria-labelledby="external-processing-modal-title">
                <SectionHeading
                  id="external-processing-modal-title"
                  title="Modal"
                  description="Run sync analysis on cloud GPUs in your own Modal workspace. Echora deploys its analysis there and stores the models in a Modal Volume."
                />
                <SettingsNotice tone="warning" title="Audio and lyrics leave this server">
                  Syncs on Modal upload each track&rsquo;s audio and lyrics to your Modal workspace
                  for processing. Uploaded audio is deleted when its batch finishes. GPU time is
                  billed to your Modal account.
                </SettingsNotice>
                <Toggle
                  label="Enable Modal"
                  description="Offer Modal as a location when starting a sync."
                  checked={value.enabled}
                  onChange={(enabled) => update({ enabled })}
                  disabled={saving}
                />
                <SettingRow
                  label="Token ID"
                  htmlFor="modal-token-id"
                  description={
                    <span>
                      Create a token in Modal under Settings → API Tokens. It starts with ak-.
                    </span>
                  }
                >
                  <Input
                    id="modal-token-id"
                    autoComplete="off"
                    spellCheck={false}
                    maxLength={200}
                    required={value.enabled}
                    value={value.token_id}
                    placeholder="ak-…"
                    onChange={(event) => update({ token_id: event.target.value })}
                  />
                </SettingRow>
                <SettingRow
                  label="Token secret"
                  htmlFor="modal-token-secret"
                  description="Stored encrypted. Leave blank to keep the stored secret, or enter a replacement."
                >
                  <div className={styles.keyControls}>
                    <Input
                      id="modal-token-secret"
                      type="password"
                      autoComplete="new-password"
                      maxLength={1000}
                      disabled={clearSecret}
                      value={secret}
                      placeholder={value.has_secret && !clearSecret ? "••••••••••••••••" : "as-…"}
                      onChange={(event) => setSecret(event.target.value)}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      className="shrink-0"
                      aria-pressed={clearSecret}
                      onClick={() => {
                        setClearSecret((current) => !current);
                        setSecret("");
                      }}
                    >
                      {clearSecret ? "Cancel removal" : "Clear on save"}
                    </Button>
                  </div>
                </SettingRow>
                <SettingRow
                  label="GPU"
                  htmlFor="modal-gpu"
                  description="Changing the GPU redeploys Echora before the next Modal sync."
                >
                  <Choice id="modal-gpu" value={value.gpu} onChange={(gpu) => update({ gpu })}>
                    {value.gpu_types.map((gpu) => (
                      <ChoiceItem key={gpu} value={gpu}>
                        {gpu} · {gpuHints[gpu] ?? ""}
                      </ChoiceItem>
                    ))}
                  </Choice>
                </SettingRow>
                <SettingRow
                  label="Preselected location"
                  htmlFor="modal-default"
                  description="Where new syncs run unless changed when starting. Jobs started elsewhere, such as backfills, use it too."
                >
                  <Choice
                    id="modal-default"
                    value={value.default_compute}
                    onChange={(choice) =>
                      update({ default_compute: choice as Configuration["default_compute"] })
                    }
                  >
                    <ChoiceItem value="local">This server</ChoiceItem>
                    <ChoiceItem value="modal">Modal</ChoiceItem>
                  </Choice>
                </SettingRow>
                <SettingRow
                  label="Hugging Face token"
                  description="Motion artwork on Modal downloads the gated LTX-2.5 model with the token saved in Analysis models."
                >
                  <span className="text-[13px]">{value.has_hf_token ? "Saved" : "Not saved"}</span>
                </SettingRow>
                <Toggle
                  label="Let all users choose Modal"
                  description="Off: only administrators can run syncs on Modal."
                  checked={value.allow_users}
                  onChange={(allow_users) => update({ allow_users })}
                  disabled={saving}
                />
              </section>
              <div className={layout.actions}>
                <Button type="submit" loading={saving}>
                  {saving ? "Checking token…" : "Save Modal settings"}
                </Button>
              </div>
            </fieldset>

            <section aria-labelledby="external-processing-status-title" aria-live="polite">
              <SectionHeading
                id="external-processing-status-title"
                title="Workspace"
                description="Every Modal sync checks the deployment first and redeploys or downloads models when Echora or its model settings change."
              />
              <SettingRow label="Workspace">
                <span className="text-[13px]">{value.workspace ?? "No token verified"}</span>
              </SettingRow>
              <SettingRow
                label="Status"
                description={`Last checked ${formatTime(value.checked_at)}. Image ${value.image}.`}
              >
                <span className="text-[13px] font-medium">
                  {setupActive ? "Preparing" : statusText[value.status]}
                </span>
              </SettingRow>
              {setupActive && setup && (
                <div className="grid gap-2 py-3">
                  <p className="text-[13px] text-muted-foreground">
                    {setup.message || "Starting preparation…"}
                  </p>
                  <div
                    role="progressbar"
                    aria-label="Modal preparation progress"
                    aria-valuemin={progress === null ? undefined : 0}
                    aria-valuemax={progress === null ? undefined : 100}
                    aria-valuenow={progress ?? undefined}
                    className="h-1 overflow-hidden bg-raised"
                  >
                    <div
                      className={`h-full bg-primary transition-[width] ${progress === null ? "animate-pulse" : ""}`}
                      style={{
                        width: progress === null ? "100%" : `${progress}%`,
                        opacity: progress === null ? 0.5 : 1,
                      }}
                    />
                  </div>
                </div>
              )}
              {!setupActive && value.status === "failed" && value.status_detail && (
                <Notice tone="error" title="Modal preparation failed">
                  {value.status_detail}
                </Notice>
              )}
              <SettingRow
                label="Prepare Modal"
                description={
                  unsaved
                    ? "Save your changes first: preparing uses the saved settings."
                    : "Deploy Echora's analysis and download its models (about 20 GB, plus 44 GB when motion artwork runs on Modal). The first Modal sync does this anyway; preparing ahead makes that sync start faster."
                }
              >
                <Button
                  type="button"
                  variant="outline"
                  loading={starting}
                  disabled={unsaved || !saved?.enabled || !saved?.has_secret || setupActive}
                  onClick={prepare}
                >
                  {setupActive
                    ? "Preparing…"
                    : value.status === "ready"
                      ? "Check again"
                      : "Prepare Modal"}
                </Button>
              </SettingRow>
            </section>
          </div>
        )
      )}
      {!loading && !value && (
        <Notice
          tone="error"
          title="Could not load External processing settings"
          action={
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setLoading(true);
                setLoadError("");
                setAttempt((current) => current + 1);
              }}
            >
              Try again
            </Button>
          }
        >
          {loadError}
        </Notice>
      )}
    </form>
  );
}
