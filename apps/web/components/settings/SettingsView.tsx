"use client";

import { setUrl, useSubPath } from "../shell/urlState";
import {
  Palette,
  Sparkles,
  BrainCircuit,
  Clapperboard,
  Cloud,
  Clock3,
  Radio,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  UserRound,
} from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import {
  defaultPlaybackPreferences,
  PlaybackPreferences,
  readPlaybackPreferences,
  writePlaybackPreferences,
} from "../player/playbackPreferences";
import { SectionHeading, Choice, ChoiceItem, LocationToggles, Toggle } from "./Presentation";

import { Button } from "../ui/button";
import { SettingsNotice } from "./SettingsNotice";
import { RangeControl } from "./RangeControl";
import { SettingRow } from "./SettingRow";
import { SidePanel } from "../layout/side-panel";
import { Pane } from "../layout/pane";
import { Badge } from "../ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { cn } from "@/lib/utils";
import styles from "./SettingsView.module.css";
import { Input } from "../ui/input";
import { LoadingState } from "../ui/spinner";
import ExternalAISettings from "./ExternalAISettings";
import ExternalProcessingSettings from "./ExternalProcessingSettings";
import { toast } from "sonner";
import { Notice } from "../ui/notice";
import NavidromePluginSettings from "./NavidromePluginSettings";
import MotionArtworkSettings from "./MotionArtworkSettings";

type Feature = "transcription" | "karaoke" | "hum";
type Location = "local" | "modal";
const featureNames: Record<Feature, string> = {
  transcription: "AI lyrics generation",
  karaoke: "Karaoke timing",
  hum: "Query by humming",
};

const allTabs = [
  "server",
  "lastfm",
  "playback",
  "models",
  "external-ai",
  "external-processing",
  "motion-artwork",
  "appearance",
  "timezone",
  "account",
  "oidc",
] as const;
type Tab = (typeof allTabs)[number];
type Settings = {
  profile: { username: string; email: string; display_name: string; is_admin: boolean };
  timezone: string;
  navidrome: { id: string; url: string; username: string } | null;
  lastfm: { connected: boolean; username?: string };
  models: {
    transcription_processing_enabled: boolean;
    karaoke_processing_enabled: boolean;
    hum_processing_enabled: boolean;
    transcription_modal_enabled: boolean;
    karaoke_modal_enabled: boolean;
    hum_modal_enabled: boolean;
    huggingface_account: string | null;
    has_huggingface_token: boolean;
  };
};
type OidcSettings = {
  configured: boolean;
  issuer?: string;
  require_verified_email: boolean;
  auto_provision: boolean;
  users: {
    id: string;
    email: string;
    display_name: string;
    is_admin: boolean;
    is_blocked: boolean;
  }[];
  allowed_emails: string[];
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/analysis${path}`, options);
  const body = response.status === 204 ? null : await response.json();
  if (!response.ok) throw new Error(body?.detail || "The request failed");
  return body as T;
}

const descriptions: Record<Tab, string> = {
  server: "Update the Navidrome server used for synchronization and playlist publishing.",
  lastfm: "Connect listening history for familiarity mixes and time-of-day curations.",
  playback: "Choose the stream sent by Navidrome.",
  models: "Choose which optional analysis runs in syncs on this server and in syncs on Modal.",
  "external-ai": "Manage endpoints and AI features.",
  "external-processing": "Run sync analysis on cloud GPUs in your Modal workspace.",
  "motion-artwork": "Generate looping video versions of album covers for the full-screen player.",
  appearance: "Set motion, karaoke highlights, and audio-reactive backgrounds.",
  timezone: "Listening periods use this timezone rather than the server clock.",
  account: "Your email is your fixed username. Your display name can be changed.",
  oidc: "Provider credentials come from the container environment. Manage who Echora may provision.",
};

const zones = [
  "UTC",
  "America/Los_Angeles",
  "America/Denver",
  "America/Chicago",
  "America/New_York",
  "America/Toronto",
  "America/Sao_Paulo",
  "Europe/London",
  "Europe/Paris",
  "Europe/Berlin",
  "Europe/Warsaw",
  "Africa/Johannesburg",
  "Asia/Dubai",
  "Asia/Kolkata",
  "Asia/Bangkok",
  "Asia/Shanghai",
  "Asia/Tokyo",
  "Asia/Seoul",
  "Australia/Sydney",
  "Pacific/Auckland",
];

export default function SettingsView() {
  // The section lives in the path (/settings/appearance) so it can be linked and survives reloads.
  const requestedTab = useSubPath("settings");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [serverUrl, setServerUrl] = useState("");
  const [serverUsername, setServerUsername] = useState("");
  const [serverPassword, setServerPassword] = useState("");
  const [lastfmUsername, setLastfmUsername] = useState("");
  const [lastfmKey, setLastfmKey] = useState("");
  const [timezone, setTimezone] = useState("UTC");
  const [displayName, setDisplayName] = useState("");
  const [oidc, setOidc] = useState<OidcSettings | null>(null);
  const [allowedEmail, setAllowedEmail] = useState("");
  const [busy, setBusy] = useState(false);
  // Only load failures stay on the page; action results are toasts.
  const [loadError, setLoadError] = useState("");
  const [playback, setPlayback] = useState<PlaybackPreferences>(defaultPlaybackPreferences);
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

  function apply(value: Settings) {
    setSettings(value);
    setServerUrl(value.navidrome?.url || "");
    setServerUsername(value.navidrome?.username || "");
    setLastfmUsername(value.lastfm.username || "");
    setTimezone(value.timezone);
    setDisplayName(value.profile.display_name);
    setHuggingface({
      has: value.models.has_huggingface_token,
      account: value.models.huggingface_account,
    });
    setFeatures({
      transcription: {
        local: value.models.transcription_processing_enabled,
        modal: value.models.transcription_modal_enabled,
      },
      karaoke: {
        local: value.models.karaoke_processing_enabled,
        modal: value.models.karaoke_modal_enabled,
      },
      hum: { local: value.models.hum_processing_enabled, modal: value.models.hum_modal_enabled },
    });
    setPlayback(readPlaybackPreferences());
  }
  function load() {
    api<Settings>("/settings")
      .then(apply)
      .catch((reason) => setLoadError(reason.message));
  }
  function loadOidc() {
    api<OidcSettings>("/settings/oidc")
      .then(setOidc)
      .catch((reason) =>
        toast.error("Could not load sign-in settings", { description: reason.message }),
      );
  }
  useEffect(load, []);
  function savePlayback(next: PlaybackPreferences) {
    setPlayback(next);
    writePlaybackPreferences(next);
    toast.success("Playback preferences saved", { id: "playback-preferences" });
  }
  function saveAnimationSpeed(next: PlaybackPreferences["animationSpeed"]) {
    savePlayback({ ...playback, animationSpeed: next });
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

  function fail(title: string, reason: unknown) {
    toast.error(title, { description: reason instanceof Error ? reason.message : undefined });
  }
  async function submit(action: () => Promise<unknown>, success: string) {
    setBusy(true);
    try {
      await action();
      toast.success(success);
      load();
    } catch (reason) {
      fail("Could not save settings", reason);
    } finally {
      setBusy(false);
    }
  }
  const serverCanSave = Boolean(serverUrl.trim() && serverUsername.trim() && serverPassword);
  const lastfmCanSave = Boolean(lastfmUsername.trim() && lastfmKey);
  const timezoneCanSave = Boolean(settings && timezone && timezone !== settings.timezone);
  const profileCanSave = Boolean(
    settings && displayName.trim() && displayName.trim() !== settings.profile.display_name,
  );

  function saveServer(event: FormEvent) {
    event.preventDefault();
    if (!serverCanSave) return;
    submit(
      () =>
        api("/settings/navidrome", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            url: serverUrl,
            username: serverUsername,
            password: serverPassword,
          }),
        }),
      "Navidrome connection verified and saved",
    ).then(() => setServerPassword(""));
  }
  function saveLastFm(event: FormEvent) {
    event.preventDefault();
    if (!lastfmCanSave) return;
    submit(
      () =>
        api("/settings/lastfm", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ username: lastfmUsername, api_key: lastfmKey }),
        }),
      "Last.fm history connection verified",
    ).then(() => setLastfmKey(""));
  }
  function saveTimezone(event: FormEvent) {
    event.preventDefault();
    if (!timezoneCanSave) return;
    submit(
      () =>
        api("/settings/timezone", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ timezone }),
        }),
      "Timezone saved",
    );
  }
  function saveProfile(event: FormEvent) {
    event.preventDefault();
    if (!profileCanSave) return;
    submit(
      () =>
        api("/settings/profile", {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ display_name: displayName }),
        }),
      "Profile saved",
    );
  }
  function updateOidc(action: () => Promise<unknown>, success: string) {
    submit(action, success).then(loadOidc);
  }
  function addAllowedEmail(event: FormEvent) {
    event.preventDefault();
    if (!allowedEmail.trim()) return;
    updateOidc(
      () =>
        api("/settings/oidc/allowed-emails", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ email: allowedEmail }),
        }),
      "User approved for OIDC sign-in",
    );
    setAllowedEmail("");
  }

  const adminTabs: Tab[] = [
    "models",
    "external-ai",
    "external-processing",
    "motion-artwork",
    "oidc",
  ];
  const knownTab = (allTabs as readonly string[]).includes(requestedTab ?? "")
    ? (requestedTab as Tab)
    : "account";
  // Admin sections fall back to Account for everyone else once we know who they are.
  const tab: Tab =
    settings && !settings.profile.is_admin && adminTabs.includes(knownTab) ? "account" : knownTab;
  // Sign-in settings load whenever that section opens, by click or by URL.
  const oidcOpen = tab === "oidc";
  // Motion artwork's location switches live in its own settings; load them with the models tab.
  const modelsOpen = tab === "models" && Boolean(settings?.profile.is_admin);
  useEffect(() => {
    if (!modelsOpen) return;
    api<{ enabled: boolean; generate_during_sync: boolean; generate_on_modal: boolean }>(
      "/settings/motion-artwork",
    )
      .then(({ enabled, generate_during_sync, generate_on_modal }) =>
        setMotionArtwork({ enabled, generate_during_sync, generate_on_modal }),
      )
      .catch(() => setMotionArtwork(null));
  }, [modelsOpen]);
  useEffect(() => {
    if (!oidcOpen) return;
    api<OidcSettings>("/settings/oidc")
      .then(setOidc)
      .catch((reason) =>
        toast.error("Could not load sign-in settings", { description: reason.message }),
      );
  }, [oidcOpen]);
  const tabs: { id: Tab; label: string; group: string; icon: typeof Server }[] = [
    { id: "account", label: "Account", group: "You", icon: UserRound },
    { id: "appearance", label: "Appearance", group: "You", icon: Palette },
    { id: "playback", label: "Playback", group: "You", icon: SlidersHorizontal },
    { id: "timezone", label: "Timezone", group: "You", icon: Clock3 },
    { id: "server", label: "Music server", group: "Connections", icon: Server },
    { id: "lastfm", label: "Listening history", group: "Connections", icon: Radio },
    ...(settings?.profile.is_admin
      ? [
          {
            id: "models" as Tab,
            label: "Analysis models",
            group: "Administration",
            icon: BrainCircuit,
          },
          {
            id: "external-ai" as Tab,
            label: "External AI",
            group: "Administration",
            icon: Sparkles,
          },
          {
            id: "external-processing" as Tab,
            label: "External processing",
            group: "Administration",
            icon: Cloud,
          },
          {
            id: "motion-artwork" as Tab,
            label: "Motion artwork",
            group: "Administration",
            icon: Clapperboard,
          },
          {
            id: "oidc" as Tab,
            label: "Sign-in & users",
            group: "Administration",
            icon: ShieldCheck,
          },
        ]
      : []),
  ];
  const groups = Array.from(new Set(tabs.map((item) => item.group)));
  const current = tabs.find((item) => item.id === tab);
  function selectTab(next: Tab) {
    setUrl(`/settings/${next}`);
  }
  const navigation = (
    <SidePanel title="Settings">
      <nav aria-label="Settings sections" className="grid gap-6">
        {groups.map((group) => (
          <div key={group}>
            <h3 className="mb-1.5 px-3 text-xs font-medium text-subtle-foreground">{group}</h3>
            <ul className="space-y-0.5">
              {tabs
                .filter((item) => item.group === group)
                .map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      onClick={() => selectTab(item.id)}
                      aria-current={tab === item.id ? "page" : undefined}
                      className={cn(
                        "relative flex h-9 w-full items-center gap-3 px-3 text-left text-[13px] font-medium transition-colors",
                        tab === item.id
                          ? "bg-raised text-foreground before:absolute before:inset-y-2 before:left-0 before:w-0.5 before:bg-primary"
                          : "text-muted-foreground hover:bg-surface hover:text-foreground",
                      )}
                    >
                      <item.icon className={cn("size-4", tab === item.id && "text-primary")} />
                      {item.label}
                    </button>
                  </li>
                ))}
            </ul>
          </div>
        ))}
      </nav>
    </SidePanel>
  );
  return (
    <>
      <div className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)] md:grid-cols-[260px_minmax(0,1fr)]">
        <div className="hidden min-h-0 border-r border-border bg-rail md:block">{navigation}</div>
        <Pane
          className="bg-workspace"
          label={current?.label || "Settings"}
          title={current?.label || "Settings"}
          subtitle={descriptions[tab]}
          actions={
            <>
              {adminTabs.includes(tab) && (
                <Badge
                  variant="outline"
                  className="h-7 gap-1.5 text-xs"
                  title="Changes in this section apply to every Echora user"
                >
                  <ShieldCheck className="size-3.5 text-primary" />
                  Admin · applies to everyone
                </Badge>
              )}
              <Select value={tab} onValueChange={(value) => selectTab(value as Tab)}>
                <SelectTrigger aria-label="Settings section" className="w-44 md:hidden">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {tabs.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          }
        >
          <section key={tab} className={`motion-enter ${styles.form}`}>
            {loadError && !settings ? (
              <Notice
                tone="error"
                title="Could not load settings"
                action={
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setLoadError("");
                      load();
                    }}
                  >
                    Try again
                  </Button>
                }
              >
                {loadError}
              </Notice>
            ) : !settings ? (
              <LoadingState label="Loading settings…" />
            ) : (
              <>
                {tab === "external-ai" && settings?.profile.is_admin && <ExternalAISettings />}
                {tab === "external-processing" && settings?.profile.is_admin && (
                  <ExternalProcessingSettings />
                )}
                {tab === "motion-artwork" && settings?.profile.is_admin && (
                  <MotionArtworkSettings connectionId={settings.navidrome?.id ?? null} />
                )}
                {tab === "server" && (
                  <form onSubmit={saveServer}>
                    <SectionHeading title="Connection credentials" />
                    <SettingRow
                      label={<>Server URL</>}
                      htmlFor="settings-field-1"
                      description="The URL of your Navidrome server, including https://."
                    >
                      <Input
                        id="settings-field-1"
                        type="url"
                        required
                        value={serverUrl}
                        onChange={(event) => setServerUrl(event.target.value)}
                      />
                    </SettingRow>
                    <SettingRow
                      label={<>Username</>}
                      htmlFor="settings-field-2"
                      description="Your Navidrome login, not your Echora account."
                    >
                      <Input
                        id="settings-field-2"
                        required
                        value={serverUsername}
                        onChange={(event) => setServerUsername(event.target.value)}
                      />
                    </SettingRow>
                    <SettingRow
                      label={<>Password</>}
                      htmlFor="settings-field-3"
                      description="Required to verify access to your collection."
                    >
                      <Input
                        id="settings-field-3"
                        type="password"
                        required
                        value={serverPassword}
                        onChange={(event) => setServerPassword(event.target.value)}
                        placeholder="Required to verify changes"
                      />
                    </SettingRow>
                    <div className={styles.actions}>
                      <Button loading={busy} disabled={!serverCanSave}>
                        Verify and save
                      </Button>
                    </div>
                  </form>
                )}
                {tab === "server" && (
                  <NavidromePluginSettings connectionId={settings?.navidrome?.id ?? null} />
                )}
                {tab === "lastfm" && (
                  <form onSubmit={saveLastFm}>
                    <SectionHeading title="Last.fm connection" />
                    <SettingRow
                      label="Connection"
                      description="Listening history connection status."
                    >
                      <span>{settings?.lastfm.connected ? "Connected" : "Not connected"}</span>
                      <p className="text-muted-foreground">
                        {settings?.lastfm.username || "No Last.fm user"}
                      </p>
                    </SettingRow>
                    <SettingRow
                      label={<>Last.fm username</>}
                      htmlFor="settings-field-4"
                      description="The account used for listening history."
                    >
                      <Input
                        id="settings-field-4"
                        required
                        value={lastfmUsername}
                        onChange={(event) => setLastfmUsername(event.target.value)}
                      />
                    </SettingRow>
                    <SettingRow
                      label={<>API key</>}
                      htmlFor="settings-field-5"
                      description="Your Last.fm API key is stored encrypted."
                    >
                      <Input
                        id="settings-field-5"
                        type="password"
                        required
                        value={lastfmKey}
                        onChange={(event) => setLastfmKey(event.target.value)}
                        placeholder="Stored encrypted"
                      />
                    </SettingRow>
                    <div className={styles.actions}>
                      <Button loading={busy} disabled={!lastfmCanSave}>
                        Verify and save
                      </Button>
                      {settings?.lastfm.connected && (
                        <Button
                          variant="outline"
                          type="button"
                          disabled={busy}
                          onClick={() =>
                            submit(
                              () => api("/settings/lastfm", { method: "DELETE" }),
                              "Last.fm disconnected",
                            )
                          }
                        >
                          Disconnect
                        </Button>
                      )}
                    </div>
                  </form>
                )}
                {tab === "playback" && (
                  <section>
                    <SectionHeading title="Streaming quality" />
                    <SettingRow
                      label={<>Music transcoding</>}
                      htmlFor="settings-field-6"
                      description={
                        <>Original keeps the source quality. Lower bitrates use less bandwidth.</>
                      }
                    >
                      <Choice
                        id="settings-field-6"
                        value={playback.quality}
                        onChange={(value) =>
                          savePlayback({
                            ...playback,
                            quality: value as PlaybackPreferences["quality"],
                          })
                        }
                      >
                        <ChoiceItem value="original">Original</ChoiceItem>
                        <ChoiceItem value="320">320 kbps</ChoiceItem>
                        <ChoiceItem value="120">120 kbps</ChoiceItem>
                      </Choice>
                    </SettingRow>
                  </section>
                )}
                {tab === "models" && settings?.profile.is_admin && (
                  <section>
                    <SectionHeading
                      title="Optional features"
                      description="Each sync runs on this server or on Modal, chosen when it starts. A feature runs only in syncs where it is switched on, so work that needs a GPU can be left to Modal."
                    />
                    <SettingsNotice tone="warning" title="AI-generated lyrics">
                      Generated words and timing may be inaccurate. Existing lyrics remain available
                      when processing is switched off.
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
                        <Button
                          type="submit"
                          variant="outline"
                          disabled={busy || !huggingfaceToken.trim()}
                        >
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
                )}
                {tab === "appearance" && (
                  <section>
                    <SectionHeading title="Motion and interface" />
                    <SettingRow
                      label={<>Animation speed</>}
                      htmlFor="settings-field-7"
                      description={<>This preference applies across the entire application.</>}
                    >
                      <Choice
                        id="settings-field-7"
                        value={playback.animationSpeed}
                        onChange={(value) =>
                          saveAnimationSpeed(value as PlaybackPreferences["animationSpeed"])
                        }
                      >
                        <ChoiceItem value="slow">Slow</ChoiceItem>
                        <ChoiceItem value="normal">Normal</ChoiceItem>
                        <ChoiceItem value="fast">Fast</ChoiceItem>
                      </Choice>
                    </SettingRow>
                    <SettingRow
                      label={<>Karaoke highlight style</>}
                      htmlFor="settings-field-8"
                      description={
                        <>
                          Highlight each syllable at once, fill it with a moving liquid edge, bring
                          words out of a blur (Bloom), or punch each word in like a combo hit
                          (Strike). Timing stays the same. Saved in this browser.
                        </>
                      }
                    >
                      <Choice
                        id="settings-field-8"
                        value={playback.karaokeHighlightStyle}
                        onChange={(value) =>
                          savePlayback({
                            ...playback,
                            karaokeHighlightStyle:
                              value as PlaybackPreferences["karaokeHighlightStyle"],
                          })
                        }
                      >
                        <ChoiceItem value="syllable">Syllable highlight</ChoiceItem>
                        <ChoiceItem value="lava">Lava fill</ChoiceItem>
                        <ChoiceItem value="bloom">Bloom</ChoiceItem>
                        <ChoiceItem value="strike">Strike</ChoiceItem>
                      </Choice>
                    </SettingRow>
                    <section className="pt-6">
                      <SectionHeading
                        title="Audio-reactive backdrop"
                        description="Choose how the desktop background responds to the playing track and its frequency bands."
                      />
                      <SettingRow
                        label={<>Backdrop style</>}
                        htmlFor="settings-field-9"
                        description={
                          <>
                            All styles use the playing track and its album-derived color palette.
                            Backdrops are hidden in the compact mobile layout.
                          </>
                        }
                      >
                        <Choice
                          id="settings-field-9"
                          value={playback.backdropPreset}
                          onChange={(value) =>
                            savePlayback({
                              ...playback,
                              backdropPreset: value as PlaybackPreferences["backdropPreset"],
                            })
                          }
                        >
                          <ChoiceItem value="waves">PS3 waves</ChoiceItem>
                          <ChoiceItem value="oscilloscope">Oscilloscope</ChoiceItem>
                          <ChoiceItem value="void">Void tunnel</ChoiceItem>
                          <ChoiceItem value="curtain">Digital curtain</ChoiceItem>
                          <ChoiceItem value="ascii">ASCII dance</ChoiceItem>
                          <ChoiceItem value="roots">Living roots</ChoiceItem>
                          <ChoiceItem value="lightningfall">Lightning Fall</ChoiceItem>
                          <ChoiceItem value="meshgrid">Mesh Grid</ChoiceItem>
                          <ChoiceItem value="clouds">Storm clouds</ChoiceItem>
                          <ChoiceItem value="waterdrops">Water drops</ChoiceItem>
                        </Choice>
                        {playback.backdropPreset === "waterdrops" && (
                          <SettingsNotice tone="info" title="Backdrop behavior">
                            Music drives invisible drop impacts on an album-colored water surface.
                            Bass sets impact size, vocals change drop velocity and ripple spread,
                            and treble changes surface tension and damping. Ripples overlap and
                            interfere, then settle when playback pauses.
                          </SettingsNotice>
                        )}
                        {playback.backdropPreset === "meshgrid" && (
                          <SettingsNotice tone="info" title="Backdrop behavior">
                            A rippling triangular landscape in album-art colors. Bass raises the
                            waves, vocals roughen the mesh, and treble lights the edges.
                          </SettingsNotice>
                        )}
                        {playback.backdropPreset === "lightningfall" && (
                          <SettingsNotice tone="info" title="Backdrop behavior">
                            Dense falling trails in album-art colors. Bass and tempo drive the flow;
                            individual frequency bands subtly vary trail length, thickness, and
                            glow. Pausing dims the trails.
                          </SettingsNotice>
                        )}
                        {playback.backdropPreset === "clouds" && (
                          <SettingsNotice tone="warning" title="Flashing effects">
                            Full-background smoke. Bass drives billowing, vocals drive cloud color,
                            and treble lights cloud edges and soft internal lightning. Attacks in
                            any band can trigger branching strikes. Drift follows estimated tempo.
                            Strikes are at least 0.8 seconds apart. Contains flashes. Set all three
                            sensitivity sliders to zero for no lightning, or turn off Animated
                            backdrop.
                          </SettingsNotice>
                        )}
                      </SettingRow>
                      <SettingRow
                        label={<>Wave frame rate</>}
                        htmlFor="settings-field-10"
                        description={
                          <>
                            Stored in this browser so each device can use an appropriate rendering
                            load.
                          </>
                        }
                      >
                        <Choice
                          id="settings-field-10"
                          value={playback.waveFrameRate}
                          onChange={(value) =>
                            savePlayback({
                              ...playback,
                              waveFrameRate: value as PlaybackPreferences["waveFrameRate"],
                            })
                          }
                        >
                          <ChoiceItem value="30">30 FPS</ChoiceItem>
                          <ChoiceItem value="60">60 FPS</ChoiceItem>
                          <ChoiceItem value="uncapped">Uncapped</ChoiceItem>
                        </Choice>
                      </SettingRow>
                      <Toggle
                        label="Animated backdrop"
                        checked={playback.wavesEnabled}
                        disabled={busy}
                        onChange={() =>
                          savePlayback({ ...playback, wavesEnabled: !playback.wavesEnabled })
                        }
                      />
                      <div>
                        <SettingRow
                          label={
                            <>
                              <b>Backdrop opacity</b>
                              <output className="font-normal tabular-nums">
                                {Math.round(playback.backdropOpacity * 100)}%
                              </output>
                            </>
                          }
                          htmlFor="settings-field-11"
                        >
                          <RangeControl
                            id="settings-field-11"
                            aria-label="Backdrop opacity"
                            min="0"
                            max="1"
                            step="0.05"
                            value={playback.backdropOpacity}
                            onChange={(event) =>
                              savePlayback({
                                ...playback,
                                backdropOpacity: Number(event.target.value),
                              })
                            }
                          />
                        </SettingRow>
                      </div>
                      <div>
                        {(
                          [
                            ["bassReactivity", "Bass"],
                            ["vocalReactivity", "Vocals"],
                            ["trebleReactivity", "Treble"],
                          ] as const
                        ).map(([key, label]) => (
                          <SettingRow
                            key={key}
                            label={
                              <>
                                <b>{label}</b>
                                <output className="font-normal tabular-nums">
                                  {Math.round(playback[key] * 100)}%
                                </output>
                              </>
                            }
                            htmlFor={`settings-${key}`}
                          >
                            <RangeControl
                              id={`settings-${key}`}
                              disabled={!playback.wavesEnabled}
                              min="0"
                              max="2"
                              step="0.05"
                              value={playback[key]}
                              onChange={(event) =>
                                savePlayback({ ...playback, [key]: Number(event.target.value) })
                              }
                            />
                          </SettingRow>
                        ))}
                      </div>
                    </section>
                  </section>
                )}
                {tab === "timezone" && (
                  <form onSubmit={saveTimezone}>
                    <SectionHeading title="Listening periods" />
                    <SettingRow
                      label={<>IANA timezone</>}
                      htmlFor="settings-field-13"
                      description="Use your local timezone for listening periods."
                    >
                      <Choice
                        id="settings-field-13"
                        value={timezone}
                        onChange={(value) => setTimezone(value)}
                      >
                        {!zones.includes(timezone) && (
                          <ChoiceItem value={timezone}>{timezone}</ChoiceItem>
                        )}
                        {zones.map((zone) => (
                          <ChoiceItem key={zone} value={zone}>
                            {zone}
                          </ChoiceItem>
                        ))}
                      </Choice>
                    </SettingRow>
                    <div className={styles.actions}>
                      <Button loading={busy} disabled={!timezoneCanSave}>
                        Save timezone
                      </Button>
                    </div>
                  </form>
                )}
                {tab === "account" && (
                  <div className="space-y-5">
                    <form onSubmit={saveProfile}>
                      <SectionHeading title="Profile details" />
                      <SettingRow
                        label={<>Email and username</>}
                        htmlFor="settings-field-14"
                        description="Your email is your fixed username."
                      >
                        <Input
                          id="settings-field-14"
                          disabled
                          value={settings?.profile.email || settings?.profile.username || ""}
                          readOnly
                        />
                      </SettingRow>
                      <SettingRow
                        label={<>Display name</>}
                        htmlFor="settings-field-15"
                        description="The name shown in your account menu."
                      >
                        <Input
                          id="settings-field-15"
                          required
                          value={displayName}
                          onChange={(event) => setDisplayName(event.target.value)}
                        />
                      </SettingRow>
                      <div className={styles.actions}>
                        <Button loading={busy} disabled={!profileCanSave}>
                          Save profile
                        </Button>
                      </div>
                    </form>
                  </div>
                )}
                {tab === "oidc" && settings?.profile.is_admin && (
                  <div>
                    {!oidc && <LoadingState label="Loading sign-in settings…" />}
                    {oidc && (
                      <>
                        <SectionHeading title="Provider and provisioning" />
                        <SettingRow label="Issuer">
                          <span className="break-all">{oidc.issuer || "Not configured"}</span>
                        </SettingRow>
                        <SettingRow label="Verified email required">
                          {oidc.require_verified_email ? "Yes" : "No"}
                        </SettingRow>
                        <Toggle
                          label="Automatically provision new OIDC users"
                          checked={oidc.auto_provision}
                          disabled={busy}
                          onChange={() =>
                            updateOidc(
                              () =>
                                api("/settings/oidc/policy", {
                                  method: "PUT",
                                  headers: { "content-type": "application/json" },
                                  body: JSON.stringify({ auto_provision: !oidc.auto_provision }),
                                }),
                              oidc.auto_provision
                                ? "Automatic provisioning disabled"
                                : "Automatic provisioning enabled",
                            )
                          }
                        />
                        <form className={styles.group} onSubmit={addAllowedEmail}>
                          <SectionHeading title="Sign-in approvals" />
                          <SettingRow
                            label={<>Explicitly approve email</>}
                            htmlFor="settings-field-16"
                            description="Allow this email address to sign in using OIDC."
                          >
                            <Input
                              id="settings-field-16"
                              type="email"
                              required
                              value={allowedEmail}
                              onChange={(event) => setAllowedEmail(event.target.value)}
                              placeholder="person@example.com"
                            />
                          </SettingRow>
                          <div className={styles.actions}>
                            <Button loading={busy} disabled={!allowedEmail.trim()}>
                              Add user
                            </Button>
                          </div>
                        </form>
                        {oidc.allowed_emails.length > 0 && (
                          <section className={styles.group}>
                            <SectionHeading
                              title="Approved emails"
                              count={oidc.allowed_emails.length}
                            />
                            {oidc.allowed_emails.map((email) => (
                              <article className={styles.identity} key={email}>
                                <span className={styles.identityText}>{email}</span>
                                <Button
                                  variant="outline"
                                  aria-label={`Remove approval for ${email}`}
                                  disabled={busy}
                                  onClick={() =>
                                    updateOidc(
                                      () =>
                                        api(
                                          `/settings/oidc/allowed-emails/${encodeURIComponent(email)}`,
                                          { method: "DELETE" },
                                        ),
                                      "Approval removed",
                                    )
                                  }
                                >
                                  Remove
                                </Button>
                              </article>
                            ))}
                          </section>
                        )}
                        <section className={styles.group}>
                          <SectionHeading title="Users" count={oidc.users.length} />
                          {oidc.users.map((item) => (
                            <article className={styles.identity} key={item.id}>
                              <div className={styles.identityText}>
                                <strong>{item.display_name}</strong>
                                <small>{item.email}</small>
                              </div>
                              <div className={styles.identityActions}>
                                <Button
                                  aria-label={`${item.is_admin ? "Remove administrator from" : "Make administrator:"} ${item.email}`}
                                  variant={item.is_admin ? "secondary" : "outline"}
                                  disabled={busy || item.email === settings.profile.email}
                                  onClick={() =>
                                    updateOidc(
                                      () =>
                                        api(`/settings/oidc/users/${item.id}`, {
                                          method: "PATCH",
                                          headers: { "content-type": "application/json" },
                                          body: JSON.stringify({ is_admin: !item.is_admin }),
                                        }),
                                      item.is_admin
                                        ? "Administrator removed"
                                        : "Administrator granted",
                                    )
                                  }
                                >
                                  {item.is_admin ? "Admin" : "User"}
                                </Button>
                                <Button
                                  aria-label={`${item.is_blocked ? "Unblock" : "Block"} ${item.email}`}
                                  variant={item.is_blocked ? "destructive" : "outline"}
                                  disabled={busy || item.email === settings.profile.email}
                                  onClick={() =>
                                    updateOidc(
                                      () =>
                                        api(`/settings/oidc/users/${item.id}`, {
                                          method: "PATCH",
                                          headers: { "content-type": "application/json" },
                                          body: JSON.stringify({ is_blocked: !item.is_blocked }),
                                        }),
                                      item.is_blocked ? "User unblocked" : "User blocked",
                                    )
                                  }
                                >
                                  {item.is_blocked ? "Unblock" : "Block"}
                                </Button>
                              </div>
                            </article>
                          ))}
                        </section>
                      </>
                    )}
                  </div>
                )}
              </>
            )}
          </section>
        </Pane>
      </div>
    </>
  );
}
