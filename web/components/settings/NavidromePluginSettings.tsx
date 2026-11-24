"use client";

import { useEffect, useState } from "react";
import { Copy } from "lucide-react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import { Notice } from "../ui/notice";
import { SectionHeading, Choice, ChoiceItem, Toggle } from "./Presentation";
import { SettingRow } from "./SettingRow";
import { RangeControl } from "./RangeControl";

type Key = {
  id: string;
  label: string;
  prefix: string;
  expires_at: string;
  last_used_at: string | null;
  status: "active" | "expired" | "revoked";
};
type Profile = {
  enabled: boolean;
  connection_id: string | null;
  musical_weight: number;
  musical_semantic_weight: number;
  missing_lyrics: "audio" | "exclude";
  max_per_artist: number;
  key_expiry_days: number;
  serve_lyrics: boolean;
  include_translations: boolean;
  lyrics_format: "ttml" | "lrc";
};
type Settings = Profile & { keys: Key[]; lyrics_paused: boolean };
type GeneratedKey = { id: string; secret: string; expires_at: string };
const initial: Profile = {
  enabled: false,
  connection_id: null,
  musical_weight: 80,
  musical_semantic_weight: 70,
  missing_lyrics: "audio",
  max_per_artist: 2,
  key_expiry_days: 90,
  serve_lyrics: true,
  include_translations: true,
  lyrics_format: "ttml",
};
const endpoint = "/analysis/settings/integrations/navidrome";

async function request<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const response = await fetch(endpoint + path, {
    method,
    cache: "no-store",
    ...(body === undefined
      ? {}
      : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  });
  const result = response.status === 204 ? null : await response.json();
  if (!response.ok)
    throw new Error(
      typeof result?.detail === "string"
        ? result.detail
        : "Could not update the Navidrome plugin integration",
    );
  return result;
}

export default function NavidromePluginSettings({ connectionId }: { connectionId: string | null }) {
  const [profile, setProfile] = useState<Profile>(initial);
  const [keys, setKeys] = useState<Key[]>([]);
  const [generated, setGenerated] = useState<GeneratedKey | null>(null);
  const [label, setLabel] = useState("Navidrome");
  const [apiUrl, setApiUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [lyricsPaused, setLyricsPaused] = useState(false);
  const [loadError, setLoadError] = useState("");

  async function load(refreshProfile = true) {
    const { keys: storedKeys, lyrics_paused, ...storedProfile } = await request<Settings>("");
    if (refreshProfile) setProfile(storedProfile);
    setKeys(storedKeys);
    setLoaded(true);
    setGenerated((current) =>
      current &&
      storedProfile.enabled &&
      storedProfile.connection_id === connectionId &&
      storedKeys.some((key) => key.id === current.id && key.status === "active")
        ? current
        : null,
    );
    setLyricsPaused(lyrics_paused);
    return storedProfile;
  }
  useEffect(() => {
    let cancelled = false;
    request<Settings>("")
      .then(({ keys: storedKeys, lyrics_paused, ...storedProfile }) => {
        if (cancelled) return;
        setProfile(storedProfile);
        setKeys(storedKeys);
        setLoaded(true);
        setLyricsPaused(lyrics_paused);
        setApiUrl(`${window.location.origin}/analysis/integrations/navidrome/v1`);
      })
      .catch((reason) => {
        if (!cancelled) setLoadError(reason.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    let cancelled = false;
    const refreshStatus = () => {
      if (document.hidden) return;
      request<Settings>("")
        .then((value) => {
          if (!cancelled) {
            setLyricsPaused(value.lyrics_paused);
            setKeys(value.keys);
            setGenerated((current) =>
              current &&
              value.enabled &&
              value.connection_id === connectionId &&
              value.keys.some((key) => key.id === current.id && key.status === "active")
                ? current
                : null,
            );
          }
        })
        .catch(() => {
          /* Keep the last known state; the backend always enforces the pause. */
        });
    };
    const timer = window.setInterval(refreshStatus, 10000);
    document.addEventListener("visibilitychange", refreshStatus);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refreshStatus);
    };
  }, [connectionId]);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    try {
      await action();
    } catch (reason) {
      toast.error("Could not update the plugin", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  }
  function save(enabled = profile.enabled) {
    return run(async () => {
      const result = await request<{ generated_key: GeneratedKey | null }>("", "PUT", {
        ...profile,
        enabled,
        connection_id: connectionId,
      });
      if (!enabled || profile.connection_id !== connectionId) setGenerated(null);
      if (result.generated_key) setGenerated(result.generated_key);
      await load();
      toast.success(enabled ? "Plugin settings saved" : "Plugin integration disabled");
    });
  }
  function update<K extends keyof Profile>(key: K, value: Profile[K]) {
    setProfile((current) => ({ ...current, [key]: value }));
  }
  const disabled = busy || !loaded;
  const lyricsOff = !profile.serve_lyrics || lyricsPaused;
  const activeKeys = keys.filter((key) => key.status === "active").length;
  const keyExpiryValid =
    Number.isInteger(profile.key_expiry_days) &&
    profile.key_expiry_days >= 1 &&
    profile.key_expiry_days <= 3650;
  async function copyKey(secret: string) {
    await navigator.clipboard.writeText(secret);
    toast.success("API key copied");
  }
  async function generateKey() {
    const next = await request<GeneratedKey>("/keys", "POST", {
      label,
      expiry_days: profile.key_expiry_days,
    });
    setGenerated(next);
    await load(false);
    toast.success("API key generated");
  }
  async function revokeKey(id: string) {
    await request(`/keys/${id}`, "DELETE");
    setGenerated((current) => (current?.id === id ? null : current));
    await load(false);
    toast.success("API key revoked");
  }

  return (
    <section className="mt-10 space-y-8" aria-label="Echora Navidrome plugin">
      <div>
        <SectionHeading
          title="Echora Navidrome plugin"
          description="Sonic discovery and enriched lyrics in Navidrome, using this account’s library and preferences."
        />
        {loadError && (
          <Notice tone="error" title="Could not load plugin settings" className="my-5">
            {loadError}
          </Notice>
        )}
        {profile.connection_id && connectionId !== profile.connection_id && (
          <Notice tone="warning" title="Connected server changed" className="my-5">
            Saving these settings binds the integration to the current connection and revokes its
            old keys.
          </Notice>
        )}
        <Toggle
          label="Plugin access"
          description="Connect Navidrome above, then enable this to generate your first key. Everyone using the key in Navidrome receives this account’s results."
          checked={profile.enabled}
          disabled={disabled || (!connectionId && !profile.enabled)}
          onChange={(next) => save(next)}
        />
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          save();
        }}
        className="space-y-8"
      >
        <fieldset disabled={disabled} className="min-w-0">
          <SectionHeading title="Discovery" />
          <SettingRow
            label={
              <>
                <b>Musical / lyrical balance</b>
                <output className="font-normal tabular-nums">{profile.musical_weight}%</output>
              </>
            }
            htmlFor="plugin-musical"
            description={`${profile.musical_weight}% musical · ${100 - profile.musical_weight}% lyrical`}
          >
            <RangeControl
              id="plugin-musical"
              aria-label="Musical match percentage"
              min="0"
              max="100"
              step="1"
              value={profile.musical_weight}
              onChange={(event) => update("musical_weight", Number(event.target.value))}
            />
          </SettingRow>
          <SettingRow
            label={
              <>
                <b>Musical evidence balance</b>
                <output className="font-normal tabular-nums">
                  {profile.musical_semantic_weight}%
                </output>
              </>
            }
            htmlFor="plugin-semantic"
            description={`${profile.musical_semantic_weight}% musical character · ${100 - profile.musical_semantic_weight}% acoustic detail`}
          >
            <RangeControl
              id="plugin-semantic"
              aria-label="Musical semantics percentage"
              min="0"
              max="100"
              step="1"
              value={profile.musical_semantic_weight}
              onChange={(event) => update("musical_semantic_weight", Number(event.target.value))}
            />
          </SettingRow>
          <SettingRow
            label="Missing lyrical analysis"
            htmlFor="plugin-missing"
            description="How to rank tracks that have no lyrics analysis yet."
          >
            <Choice
              id="plugin-missing"
              value={profile.missing_lyrics}
              onChange={(value) => update("missing_lyrics", value as Profile["missing_lyrics"])}
            >
              <ChoiceItem value="audio">Use musical evidence only</ChoiceItem>
              <ChoiceItem value="exclude">Exclude these tracks</ChoiceItem>
            </Choice>
          </SettingRow>
          <SettingRow
            label="Maximum tracks per artist"
            htmlFor="plugin-per-artist"
            description="Keeps results varied."
          >
            <Input
              id="plugin-per-artist"
              type="number"
              min="1"
              max="20"
              required
              value={profile.max_per_artist}
              onChange={(event) => update("max_per_artist", Number(event.target.value))}
            />
          </SettingRow>
        </fieldset>

        <fieldset disabled={disabled} className="min-w-0">
          <SectionHeading title="Lyrics" />
          {lyricsPaused && (
            <Notice tone="info" title="Lyrics temporarily paused" className="my-5">
              Sync or lyrics jobs are running, so Navidrome uses its next configured source.{" "}
              {profile.serve_lyrics
                ? "Echora lyrics resume automatically when all jobs finish."
                : "Your saved preference is off and will remain off."}
            </Notice>
          )}
          <Toggle
            label="Provide Echora lyrics"
            checked={profile.serve_lyrics && !lyricsPaused}
            disabled={disabled || lyricsPaused}
            onChange={(next) => update("serve_lyrics", next)}
          />
          <Toggle
            label="Include available translations"
            checked={profile.include_translations}
            disabled={disabled || lyricsOff}
            onChange={(next) => update("include_translations", next)}
          />
          <SettingRow
            label="Lyrics format"
            htmlFor="plugin-format"
            description="Enhanced lyrics need a compatible Navidrome version and client. Existing transcripts and translations are served without starting new analysis."
          >
            <Choice
              id="plugin-format"
              disabled={disabled || lyricsOff}
              value={profile.lyrics_format}
              onChange={(value) => update("lyrics_format", value as Profile["lyrics_format"])}
            >
              <ChoiceItem value="ttml">Enhanced timing and translations (TTML)</ChoiceItem>
              <ChoiceItem value="lrc">Line timing and language variants (LRC)</ChoiceItem>
            </Choice>
          </SettingRow>
        </fieldset>

        <div className="flex justify-end">
          <Button loading={busy} disabled={!loaded || (!connectionId && profile.enabled)}>
            Save plugin settings
          </Button>
        </div>
      </form>

      <section aria-label="API keys" className="min-w-0">
        <SectionHeading
          title="API keys"
          count={activeKeys}
          description="Install echora.ndp in Navidrome, enable it, and set its URL and API key. Put echora first in Agents for discovery and in LyricsPriority for lyrics."
        />
        <SettingRow label="Plugin API URL" htmlFor="plugin-url">
          <Input id="plugin-url" readOnly value={apiUrl} />
        </SettingRow>
        {generated && profile.enabled && profile.connection_id === connectionId && (
          <Notice tone="success" title="Copy this key now. It is shown only once." className="my-5">
            <div className="mt-2 flex gap-2">
              <Input
                aria-label="New Navidrome API key"
                readOnly
                value={generated.secret}
                className="font-mono"
              />
              <Button
                type="button"
                variant="outline"
                onClick={() => run(() => copyKey(generated.secret))}
              >
                <Copy />
                Copy
              </Button>
              <Button type="button" variant="ghost" onClick={() => setGenerated(null)}>
                Dismiss
              </Button>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Expires {new Date(generated.expires_at).toLocaleDateString()}
            </p>
          </Notice>
        )}
        <fieldset disabled={disabled} className="min-w-0">
          <SettingRow label="New key label" htmlFor="plugin-key-label">
            <Input
              id="plugin-key-label"
              maxLength={100}
              value={label}
              onChange={(event) => setLabel(event.target.value)}
            />
          </SettingRow>
          <SettingRow
            label="Key expiry (days)"
            htmlFor="plugin-key-expiry"
            description="Used for new keys. Save plugin settings to keep this default. Existing keys keep their expiry dates."
          >
            <Input
              id="plugin-key-expiry"
              type="number"
              min="1"
              max="3650"
              required
              value={profile.key_expiry_days}
              onChange={(event) => update("key_expiry_days", Number(event.target.value))}
            />
          </SettingRow>
        </fieldset>
        <div className="flex justify-end py-4">
          <Button
            type="button"
            variant="outline"
            loading={busy}
            disabled={
              !loaded || !profile.enabled || !label.trim() || generated !== null || !keyExpiryValid
            }
            onClick={() => run(generateKey)}
          >
            Generate key
          </Button>
        </div>
        {keys.length > 0 && (
          <ul className="divide-y divide-border border-y border-border">
            {keys.map((key) => (
              <li key={key.id} className="flex flex-wrap items-center gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <strong className="truncate text-[13px] font-semibold">{key.label}</strong>
                    <Badge
                      variant={key.status === "active" ? "secondary" : "outline"}
                      className="capitalize"
                    >
                      {key.status}
                    </Badge>
                  </div>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    <span className="font-mono">{key.prefix}…</span> · expires{" "}
                    {new Date(key.expires_at).toLocaleDateString()} · last used{" "}
                    {key.last_used_at ? new Date(key.last_used_at).toLocaleString() : "never"}
                  </p>
                </div>
                {key.status !== "revoked" && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={busy}
                    onClick={() => run(() => revokeKey(key.id))}
                  >
                    Revoke
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </section>
  );
}
