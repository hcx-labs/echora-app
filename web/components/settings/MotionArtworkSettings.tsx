"use client";

import { toast } from "sonner";
import { FormEvent, useCallback, useEffect, useState } from "react";
import { Notice } from "../ui/notice";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { LoadingState } from "../ui/spinner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { SectionHeading, Toggle, Choice, ChoiceItem, LocationToggles } from "./Presentation";
import { SettingRow } from "./SettingRow";
import { SettingsNotice } from "./SettingsNotice";
import { RangeControl } from "./RangeControl";
import styles from "./ExternalAISettings.module.css";
import layout from "./SettingsView.module.css";

type Configuration = {
  enabled: boolean;
  comfyui_url: string;
  resolution: number;
  frames: number;
  upscale: boolean;
  camera_lock: number;
  prompt_mode: "auto" | "external" | "fixed";
  instructions: string;
  fixed_prompt: string;
  seed: number | null;
  generate_during_sync: boolean;
  generate_on_modal: boolean;
  regenerate_outdated: boolean;
};
type Defaults = { comfyui_url: string; instructions: string };
type Status = {
  enabled: boolean;
  comfyui: {
    mode: "embedded" | "external";
    installed?: boolean;
    url?: string;
    reachable?: boolean;
    version?: string;
    device?: string;
    missing_models: string[];
  };
  external_ai: { enabled: boolean; model: string | null };
  artworks: { complete: number; failed: number; bytes: number };
  tracks_with_artwork: number;
};

const endpoint = "/analysis/settings/motion-artwork";
// Rendering time on an RTX 3090 with the int8 LTX-2.5 distilled model, for 10-second loops.
const resolutions = [
  { value: 768, label: "768 × 768", hint: "about 1 min per track" },
  { value: 1024, label: "1024 × 1024", hint: "about 2.5 min per track" },
  { value: 1536, label: "1536 × 1536", hint: "about 9 min per track, needs a 24 GB GPU" },
  { value: 512, label: "512 × 512", hint: "fastest, soft detail" },
];
// Upscaling renders at these sizes and doubles them.
const upscaledResolutions = [
  {
    value: 768,
    label: "1536 × 1536",
    hint: "Rendered at 768, then upscaled. About 5 min per track, needs a 24 GB GPU.",
  },
  { value: 512, label: "1024 × 1024", hint: "Rendered at 512, then upscaled." },
];

async function request<T>(path = "", options?: RequestInit): Promise<T> {
  const response = await fetch(endpoint + path, {
    ...options,
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!response.ok) {
    if (response.status === 401) throw new Error("Your session expired. Sign in again.");
    if (response.status === 403) throw new Error("Administrator access is required.");
    if (response.status === 422) throw new Error("Check the ComfyUI URL and prompt settings.");
    throw new Error("Could not reach motion artwork settings. Try again.");
  }
  return response.json();
}

function formatBytes(bytes: number) {
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

export default function MotionArtworkSettings({ connectionId }: { connectionId: string | null }) {
  const [value, setValue] = useState<Configuration | null>(null);
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    request<Configuration & { defaults: Defaults }>("", { signal: controller.signal })
      .then(({ defaults: received, ...configuration }) => {
        if (!controller.signal.aborted) {
          setValue(configuration);
          setDefaults(received);
        }
      })
      .catch((reason) => {
        if (!controller.signal.aborted)
          setLoadError(reason instanceof Error ? reason.message : "Could not load settings.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [attempt]);

  const loadStatus = useCallback(
    (signal?: AbortSignal) =>
      request<Status>("/status", { signal }).then((next) => {
        if (!signal?.aborted) setStatus(next);
        return next;
      }),
    [],
  );
  useEffect(() => {
    const controller = new AbortController();
    loadStatus(controller.signal).catch(() => {});
    return () => controller.abort();
  }, [loadStatus, attempt]);

  function update(patch: Partial<Configuration>) {
    setValue((current) => (current ? { ...current, ...patch } : current));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!value || saving) return;
    if (value.prompt_mode === "fixed" && !value.fixed_prompt.trim()) {
      toast.error("Check these settings", {
        description: "Enter the prompt to use for every cover.",
      });
      return;
    }
    setSaving(true);
    try {
      const saved = await request<Configuration>("", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(value),
      });
      setValue(saved);
      toast.success("Motion artwork settings saved");
      loadStatus().catch(() => {});
    } catch (reason) {
      toast.error("Could not save motion artwork settings", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  async function clearLoops() {
    if (clearing) return;
    setClearing(true);
    try {
      const response = await fetch(`${endpoint}/loops`, {
        method: "DELETE",
        credentials: "same-origin",
      });
      if (response.status === 409)
        throw new Error("Wait for active syncs and imports to finish, then try again.");
      if (!response.ok) throw new Error("Could not clear the loops.");
      const result = await response.json();
      setConfirmClear(false);
      toast.success(`Cleared ${result.deleted} loops`, {
        description: "The next sync renders them again while motion artwork is on.",
      });
      loadStatus().catch(() => {});
    } catch (reason) {
      toast.error("Could not clear loops", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setClearing(false);
    }
  }

  if (loading) return <LoadingState label="Loading motion artwork settings…" />;
  if (!value || !defaults)
    return (
      <Notice
        tone="error"
        title="Could not load motion artwork settings"
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
    );

  const comfy = status?.comfyui;
  return (
    <form className={styles.sections} onSubmit={save} aria-busy={saving}>
      <fieldset className="grid min-w-0 gap-8 border-0 p-0" disabled={saving}>
        <section aria-labelledby="motion-artwork-generation">
          <SectionHeading
            id="motion-artwork-generation"
            title="Generation"
            description="A song-guided animated cover for each track, rendered with LTX-2.5 and shown in the full-screen player. The title, artist, album and available lyrics guide each prompt. The analysis worker starts ComfyUI only while a batch renders loops, then stops it."
          />
          <Toggle
            label="Enable motion artwork"
            checked={value.enabled}
            onChange={(enabled) => update({ enabled })}
            description="Off by default. Needs an NVIDIA GPU and the motion artwork models on this server, or Modal."
          />
          <LocationToggles
            label="Generate during syncs"
            description="Render loops for tracks that need one as part of syncs and imports, on this server, on Modal, or both. Syncs on Modal render on Modal's GPU and need the Hugging Face token in External processing."
            local={value.generate_during_sync}
            modal={value.generate_on_modal}
            onChange={(location, next) =>
              update(
                location === "modal" ? { generate_on_modal: next } : { generate_during_sync: next },
              )
            }
          />
          <SettingRow
            label="ComfyUI"
            description={
              !comfy
                ? undefined
                : comfy.mode === "embedded"
                  ? comfy.installed
                    ? "Built into the analysis worker. Starts only while rendering."
                    : "Not installed in this analysis image. Use the GPU image, or enter an external ComfyUI below."
                  : comfy.reachable
                    ? `External · connected${comfy.version ? ` · ComfyUI ${comfy.version}` : ""}${comfy.device ? ` · ${comfy.device}` : ""}`
                    : "External · not reachable"
            }
          >
            <span>
              {!comfy
                ? "Checking…"
                : comfy.mode === "embedded"
                  ? comfy.installed
                    ? "Built in"
                    : "Unavailable"
                  : comfy.reachable
                    ? "External"
                    : "Unreachable"}
            </span>
          </SettingRow>
          <SettingRow
            label="External ComfyUI URL"
            htmlFor="motion-artwork-url"
            description="Optional. Leave blank to use the built-in one; enter an address to render on an already running ComfyUI instead."
          >
            <Input
              id="motion-artwork-url"
              type="url"
              maxLength={500}
              value={value.comfyui_url}
              placeholder={defaults.comfyui_url || "Built in"}
              onChange={(event) => update({ comfyui_url: event.target.value })}
            />
          </SettingRow>
          {comfy && comfy.missing_models.length > 0 && (
            <SettingsNotice tone="warning" title="Models missing">
              Download them with <code>download_models --motion-artwork</code> (about 47 GB, needs a
              Hugging Face token with access to Lightricks/LTX-2.5). Missing:{" "}
              {comfy.missing_models.join(", ")}.
            </SettingsNotice>
          )}
        </section>

        <section aria-labelledby="motion-artwork-render">
          <SectionHeading
            id="motion-artwork-render"
            title="Rendering"
            description="Changes apply to loops rendered from now on. Existing loops keep playing unless you choose to regenerate them below."
          />
          <Toggle
            label="Upscale"
            checked={value.upscale}
            onChange={(upscale) =>
              update(upscale && value.resolution > 768 ? { upscale, resolution: 768 } : { upscale })
            }
            description="Render at half size, double it with the LTX-2.5 upscaler and refine the result. Sharper than rendering at the larger size directly, and faster."
          />
          <SettingRow
            label="Resolution"
            htmlFor="motion-artwork-resolution"
            description={
              (value.upscale ? upscaledResolutions : resolutions).find(
                (item) => item.value === value.resolution,
              )?.hint
            }
          >
            <Choice
              id="motion-artwork-resolution"
              value={String(value.resolution)}
              onChange={(next) => update({ resolution: Number(next) })}
            >
              {(value.upscale ? upscaledResolutions : resolutions).map((item) => (
                <ChoiceItem key={item.value} value={String(item.value)}>
                  {item.label}
                </ChoiceItem>
              ))}
            </Choice>
          </SettingRow>
          <SettingRow
            label={
              <>
                <b>Camera lock</b>
                <output className="font-normal tabular-nums">
                  {value.camera_lock === 0 ? "Off" : `${Math.round(value.camera_lock * 100)}%`}
                </output>
              </>
            }
            htmlFor="motion-artwork-camera-lock"
            description="Strength of Lightricks' static-camera LoRA. Higher holds the framing more firmly against zooms and drift, but also damps the movement of the subject."
          >
            <RangeControl
              id="motion-artwork-camera-lock"
              min="0"
              max="1"
              step="0.05"
              value={value.camera_lock}
              onChange={(event) => update({ camera_lock: Number(event.target.value) })}
            />
          </SettingRow>
          <SettingRow
            label="Loop length"
            htmlFor="motion-artwork-frames"
            description="At 24 frames per second. Every loop starts and ends on the cover, so it repeats seamlessly. 5 seconds renders in about half the time."
          >
            <Choice
              id="motion-artwork-frames"
              value={String(value.frames)}
              onChange={(next) => update({ frames: Number(next) })}
            >
              <ChoiceItem value="241">10 seconds</ChoiceItem>
              <ChoiceItem value="121">5 seconds</ChoiceItem>
            </Choice>
          </SettingRow>
          <SettingRow
            label="Seed"
            htmlFor="motion-artwork-seed"
            description="The same seed and settings reproduce the same loop. Leave blank for a new seed for every album."
          >
            <Input
              id="motion-artwork-seed"
              type="number"
              min={0}
              step={1}
              value={value.seed ?? ""}
              placeholder="Random"
              onChange={(event) =>
                update({
                  seed:
                    event.target.value === ""
                      ? null
                      : Math.max(0, Math.floor(Number(event.target.value))),
                })
              }
            />
          </SettingRow>
          <Toggle
            label="Regenerate loops made with other settings"
            checked={value.regenerate_outdated}
            onChange={(regenerate_outdated) => update({ regenerate_outdated })}
            description="Off: entire-library syncs keep existing loops and only render missing ones. On: they also render again every loop made with different rendering or prompt settings, replacing it. Each loop records the settings it was made with."
          />
        </section>

        <section aria-labelledby="motion-artwork-prompt">
          <SectionHeading
            id="motion-artwork-prompt"
            title="Prompt"
            description="What the video model is asked to animate."
          />
          <SettingRow
            label="Prompt source"
            htmlFor="motion-artwork-prompt-mode"
            description={
              value.prompt_mode === "external"
                ? `Your External AI model${status?.external_ai.model ? ` (${status.external_ai.model})` : ""} looks at each cover and writes a lively scene description. Each cover image is sent to that provider.`
                : value.prompt_mode === "auto"
                  ? "The built-in Gemma 4 E2B looks at each cover and writes a scene description, following the instructions below. Runs locally; results are plainer."
                  : "Every cover uses the same prompt. {album} and {artist} are replaced with the album's details."
            }
          >
            <Choice
              id="motion-artwork-prompt-mode"
              value={value.prompt_mode}
              onChange={(next) => update({ prompt_mode: next as Configuration["prompt_mode"] })}
            >
              <ChoiceItem value="external">Written by External AI</ChoiceItem>
              <ChoiceItem value="auto">Written by built-in Gemma</ChoiceItem>
              <ChoiceItem value="fixed">Same prompt for every cover</ChoiceItem>
            </Choice>
          </SettingRow>
          {value.prompt_mode === "external" && status && !status.external_ai.enabled && (
            <SettingsNotice tone="warning" title="External AI is off">
              Enable it and set a vision-capable model in Settings → External AI, or loops will not
              be generated.
            </SettingsNotice>
          )}
          {value.prompt_mode === "external" ? (
            <SettingRow
              label="Additional instructions"
              htmlFor="motion-artwork-instructions"
              description="Optional. Added to the built-in lively prompt-writing instructions, for example to favour a certain kind of motion."
            >
              <Textarea
                className={styles.prompt}
                id="motion-artwork-instructions"
                maxLength={4000}
                rows={4}
                value={value.instructions}
                placeholder="For example: favour gentle rocking or swaying within the artwork."
                onChange={(event) => update({ instructions: event.target.value })}
              />
            </SettingRow>
          ) : value.prompt_mode === "auto" ? (
            <SettingRow
              label="Instructions"
              htmlFor="motion-artwork-instructions"
              description={
                <>
                  Describe subject or background motion. The camera stays stationary. Leave blank
                  for the default.{" "}
                  {value.instructions && (
                    <Button
                      type="button"
                      variant="link"
                      size="sm"
                      className="h-auto p-0"
                      onClick={() => update({ instructions: "" })}
                    >
                      Reset to default
                    </Button>
                  )}
                </>
              }
            >
              <Textarea
                className={styles.prompt}
                id="motion-artwork-instructions"
                maxLength={4000}
                rows={6}
                value={value.instructions}
                placeholder={defaults.instructions}
                onChange={(event) => update({ instructions: event.target.value })}
              />
            </SettingRow>
          ) : (
            <SettingRow
              label="Prompt"
              htmlFor="motion-artwork-fixed"
              description="Describe the motion as one paragraph: the shot, how the camera moves, and what happens."
            >
              <Textarea
                className={styles.prompt}
                id="motion-artwork-fixed"
                required
                maxLength={4000}
                rows={6}
                value={value.fixed_prompt}
                placeholder="A static shot of the {album} cover. Light drifts slowly across the artwork and the camera does not move."
                onChange={(event) => update({ fixed_prompt: event.target.value })}
              />
            </SettingRow>
          )}
        </section>
        <div className={layout.actions}>
          <Button type="submit" loading={saving}>
            {saving ? "Saving…" : "Save motion artwork settings"}
          </Button>
        </div>
      </fieldset>

      <section aria-labelledby="motion-artwork-library">
        <SectionHeading
          id="motion-artwork-library"
          title="Your library"
          description="Loops are made by library syncs and imports. Every track gets its own song-guided loop, and tracks that already have a loop for the current settings are skipped. Sync progress shows on the Sync page."
        />
        {status ? (
          <>
            <SettingRow
              label="Loops rendered"
              description={
                status.artworks.failed
                  ? `${status.artworks.failed} ${status.artworks.failed === 1 ? "cover" : "covers"} failed and ${status.artworks.failed === 1 ? "is" : "are"} retried on the next sync.`
                  : undefined
              }
            >
              <span className="tabular-nums">
                {status.artworks.complete} loops · {formatBytes(status.artworks.bytes)} ·{" "}
                {status.tracks_with_artwork} of your tracks
              </span>
            </SettingRow>
          </>
        ) : (
          <LoadingState label="Checking motion artwork status…" />
        )}
        {!connectionId && (
          <SettingsNotice tone="info" title="Connect a music server">
            Motion artwork is generated for the albums in your connected Navidrome library.
          </SettingsNotice>
        )}
        <SettingRow
          label="Clear all loops"
          description="Deletes every rendered loop and its video file, for every user. Covers fall back to the still image until the next sync renders them again."
        >
          <Button
            variant="destructive"
            type="button"
            disabled={clearing || (!status?.artworks.complete && !status?.artworks.failed)}
            onClick={() => setConfirmClear(true)}
          >
            Clear all loops
          </Button>
        </SettingRow>
      </section>
      {confirmClear && (
        <Dialog
          open
          onOpenChange={(open) => {
            if (!open && !clearing) setConfirmClear(false);
          }}
        >
          <DialogContent
            showCloseButton={!clearing}
            aria-busy={clearing}
            onEscapeKeyDown={(event) => {
              if (clearing) event.preventDefault();
            }}
            onInteractOutside={(event) => {
              if (clearing) event.preventDefault();
            }}
          >
            <DialogHeader className="pr-6 text-left">
              <DialogTitle>Clear all loops?</DialogTitle>
              <DialogDescription>
                This deletes {status?.artworks.complete ?? 0} rendered loops (
                {formatBytes(status?.artworks.bytes ?? 0)}) for every user. Rendering them again
                takes about 4 minutes each. This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                disabled={clearing}
                onClick={() => setConfirmClear(false)}
              >
                Keep loops
              </Button>
              <Button type="button" variant="destructive" loading={clearing} onClick={clearLoops}>
                {clearing ? "Clearing…" : "Clear all loops"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </form>
  );
}
