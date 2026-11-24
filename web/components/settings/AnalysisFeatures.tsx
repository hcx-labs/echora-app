"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { LoadingState } from "../ui/spinner";
import { SectionHeading, LocationToggles } from "./Presentation";
import { SettingRow } from "./SettingRow";
import { SettingsNotice } from "./SettingsNotice";
import { Notice } from "../ui/notice";
import styles from "./SettingsView.module.css";

type Feature = "transcription" | "karaoke" | "hum";
type Location = "local" | "modal";
type Models = {
  transcription_processing_enabled: boolean;
  karaoke_processing_enabled: boolean;
  hum_processing_enabled: boolean;
  transcription_modal_enabled: boolean;
  karaoke_modal_enabled: boolean;
  hum_modal_enabled: boolean;
  huggingface_account: string | null;
  has_huggingface_token: boolean;
};
const featureNames: Record<Feature, string> = {
  transcription: "AI lyrics generation",
  karaoke: "Karaoke timing",
  hum: "Query by humming",
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/analysis${path}`, options);
  const body = response.status === 204 ? null : await response.json();
  if (!response.ok) throw new Error(body?.detail || "The request failed");
  return body as T;
}

/** Which optional analysis syncs run, on this server and on Modal. Admins only; used by
 * Settings → Analysis models and by onboarding. */
export default function AnalysisFeatures() {
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [busy, setBusy] = useState(false);
  // Hugging Face token for gated model downloads, shared by this server and Modal.
  const [huggingface, setHuggingface] = useState<{ has: boolean; account: string | null }>({
    has: false,
    account: null,
  });
  const [huggingfaceToken, setHuggingfaceToken] = useState("");
  const [motionArtwork, setMotionArtwork] = useState<{
    enabled: boolean;
    generate_during_sync: boolean;
    generate_on_modal: boolean;
  } | null>(null);
  // Optional features, switched on separately for syncs on this server and on Modal.
  const [features, setFeatures] = useState<Record<Feature, Record<Location, boolean>>>({
    transcription: { local: false, modal: false },
    karaoke: { local: true, modal: true },
    hum: { local: true, modal: true },
  });

  useEffect(() => {
    api<{ models: Models }>("/settings")
      .then(({ models }) => {
        setHuggingface({ has: models.has_huggingface_token, account: models.huggingface_account });
        setFeatures({
          transcription: {
            local: models.transcription_processing_enabled,
            modal: models.transcription_modal_enabled,
          },
          karaoke: {
            local: models.karaoke_processing_enabled,
            modal: models.karaoke_modal_enabled,
          },
          hum: { local: models.hum_processing_enabled, modal: models.hum_modal_enabled },
        });
        setLoaded(true);
      })
      .catch((reason) => setLoadError(reason.message));
    // Motion artwork's location switches live in its own settings; load them alongside.
    api<{ enabled: boolean; generate_during_sync: boolean; generate_on_modal: boolean }>(
      "/settings/motion-artwork",
    )
      .then(({ enabled, generate_during_sync, generate_on_modal }) =>
        setMotionArtwork({ enabled, generate_during_sync, generate_on_modal }),
      )
      .catch(() => setMotionArtwork(null));
  }, []);

  function fail(title: string, reason: unknown) {
    toast.error(title, { description: reason instanceof Error ? reason.message : undefined });
  }
  async function saveHuggingfaceToken(token: string) {
    setBusy(true);
    try {
      const result = await api<{ has_token: boolean; account: string | null }>(
        "/settings/models/huggingface",
        {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ token }),
        },
      );
      setHuggingface({ has: result.has_token, account: result.account });
      setHuggingfaceToken("");
      toast.success(result.has_token ? "Hugging Face token saved" : "Hugging Face token removed", {
        description: result.account ? `Signed in as ${result.account}.` : undefined,
      });
    } catch (reason) {
      fail("Could not save the Hugging Face token", reason);
    } finally {
      setBusy(false);
    }
  }
  async function saveMotionArtwork(location: Location, next: boolean) {
    setBusy(true);
    try {
      await api("/settings/motion-artwork/location", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ enabled: next, location }),
      });
      setMotionArtwork((current) =>
        current
          ? {
              ...current,
              [location === "modal" ? "generate_on_modal" : "generate_during_sync"]: next,
            }
          : current,
      );
      toast.success(
        `Motion artwork ${next ? "on" : "off"} for ${location === "modal" ? "Modal syncs" : "syncs on this server"}`,
        {
          description:
            next && location === "modal"
              ? "The next Modal sync downloads the motion artwork models (about 44 GB) to Modal first."
              : undefined,
        },
      );
    } catch (reason) {
      fail("Could not save motion artwork", reason);
    } finally {
      setBusy(false);
    }
  }
  async function saveFeature(feature: Feature, location: Location, next: boolean) {
    const where = location === "modal" ? "Modal syncs" : "syncs on this server";
    const name = featureNames[feature];
    setBusy(true);
    try {
      const result = await api<{ pending?: number }>(`/settings/models/${feature}`, {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ enabled: next, location }),
      });
      setFeatures((current) => ({
        ...current,
        [feature]: { ...current[feature], [location]: next },
      }));
      const pending =
        typeof result.pending === "number" ? `${result.pending} tracks are waiting. ` : "";
      toast.success(`${name} ${next ? "on" : "off"} for ${where}`, {
        description: next
          ? `${pending}The next Entire Library sync ${location === "modal" ? "on Modal" : "on this server"} processes them.`
          : `Existing results remain available. ${where[0].toUpperCase() + where.slice(1)} skip new ${name.toLowerCase()} work.`,
      });
    } catch (reason) {
      fail(`Could not save ${name.toLowerCase()}`, reason);
    } finally {
      setBusy(false);
    }
  }
  if (loadError) return <Notice tone="error">{loadError}</Notice>;
  if (!loaded) return <LoadingState label="Loading analysis settings…" />;
  return (
    <section>
      <SectionHeading
        title="Optional features"
        description="Each sync runs on this server or on Modal, chosen when it starts. A feature runs only in syncs where it is switched on, so work that needs a GPU can be left to Modal."
      />
      <SettingsNotice tone="warning" title="AI-generated lyrics">
        Generated words and timing may be inaccurate. Existing lyrics remain available when
        processing is switched off.
      </SettingsNotice>
      <LocationToggles
        label="AI lyrics generation"
        description="Use Mel-Band-Roformer and MOSS to transcribe tracks with missing lyrics. Off by default. Switching off prevents further tracks from starting."
        local={features.transcription.local}
        modal={features.transcription.modal}
        disabled={busy}
        onChange={(location, next) => saveFeature("transcription", location, next)}
      />
      <LocationToggles
        label="Karaoke timing"
        description="Generate syllable timing for tracks with synced lyrics."
        local={features.karaoke.local}
        modal={features.karaoke.modal}
        disabled={busy}
        onChange={(location, next) => saveFeature("karaoke", location, next)}
      />
      <LocationToggles
        label="Query by humming"
        description="Extract melody contours from the full mix, vocals, and accompaniment. Tracks already indexed remain searchable."
        local={features.hum.local}
        modal={features.hum.modal}
        disabled={busy}
        onChange={(location, next) => saveFeature("hum", location, next)}
      />
      {motionArtwork && (
        <LocationToggles
          label="Motion artwork"
          description={
            motionArtwork.enabled
              ? "Render looping covers with LTX-2.5. Style and quality are in Motion artwork settings."
              : "Enable motion artwork in Motion artwork settings first."
          }
          local={motionArtwork.generate_during_sync}
          modal={motionArtwork.generate_on_modal}
          disabled={busy || !motionArtwork.enabled}
          onChange={saveMotionArtwork}
        />
      )}
      <div className={styles.group}>
        <SectionHeading
          title="Always processed"
          description="Search, curations and voice filters depend on these, so every sync runs them wherever it runs."
        />
      </div>
      <SettingRow label="Audio embeddings" description="MuQ-MuLan and MERT">
        <span className="text-[12px] text-muted-foreground">Every sync</span>
      </SettingRow>
      <SettingRow label="Lyrics embeddings" description="BGE-M3">
        <span className="text-[12px] text-muted-foreground">Every sync</span>
      </SettingRow>
      <SettingRow label="Voice detection" description="Vocal presence and voice type">
        <span className="text-[12px] text-muted-foreground">Every sync</span>
      </SettingRow>
      <div className={styles.group}>
        <SectionHeading
          title="Model downloads"
          description="Some models are gated on Hugging Face, such as LTX-2.5 for motion artwork. One token serves downloads on this server and on Modal."
        />
      </div>
      <SettingRow
        label="Hugging Face token"
        htmlFor="huggingface-token"
        description={
          huggingface.has
            ? `Saved${huggingface.account ? ` for ${huggingface.account}` : ""}. Enter a new token to replace it. Accept each gated model's license with this account.`
            : "A read token from Hugging Face → Settings → Access Tokens. Accept each gated model's license with the same account. Stored encrypted."
        }
      >
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (huggingfaceToken.trim()) void saveHuggingfaceToken(huggingfaceToken);
          }}
        >
          <Input
            id="huggingface-token"
            type="password"
            autoComplete="new-password"
            maxLength={1000}
            value={huggingfaceToken}
            placeholder={huggingface.has ? "••••••••••••••••" : "hf_…"}
            onChange={(event) => setHuggingfaceToken(event.target.value)}
            disabled={busy}
          />
          <Button type="submit" variant="outline" disabled={busy || !huggingfaceToken.trim()}>
            Save
          </Button>
          {huggingface.has && (
            <Button
              type="button"
              variant="outline"
              disabled={busy}
              onClick={() => saveHuggingfaceToken("")}
            >
              Remove
            </Button>
          )}
        </form>
      </SettingRow>
    </section>
  );
}
