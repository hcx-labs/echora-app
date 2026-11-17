"use client";

import { ArrowRight, CircleCheck, Server } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import LoadingImage from "../media/LoadingImage";
import { coverArtUrl } from "../media/coverArt";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Notice } from "../ui/notice";
import { api, json, selectConnection, type Connection, type Track } from "./onboardingApi";
import { StepFrame, type StepNav } from "./StepFrame";

type Discovered = {
  connection_id: string;
  server: { version: string; url: string };
  tracks: Track[];
};

const DEFAULT_URL = "http://host.docker.internal:4533";

/** Connect a Navidrome server, with visible progress while Echora reaches it. */
export default function ServerStep({
  nav,
  connection,
  onConnected,
}: {
  nav: StepNav;
  connection: Connection | null;
  onConnected: (connection: Connection) => void;
}) {
  const [editing, setEditing] = useState(!connection);
  const [url, setUrl] = useState(connection?.url || DEFAULT_URL);
  const [username, setUsername] = useState(connection?.username || "");
  const [password, setPassword] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<{ tracks: Track[]; version?: string } | null>(null);

  // A connection saved earlier (after a reload or coming Back) shows a fresh peek at the library.
  useEffect(() => {
    if (!connection || editing || preview) return;
    let active = true;
    api<{ tracks: Track[] }>(`/navidrome/connections/${connection.id}/sample?limit=18`)
      .then((result) => active && setPreview({ tracks: result.tracks }))
      .catch(() => active && setPreview({ tracks: [] }));
    return () => {
      active = false;
    };
  }, [connection, editing, preview]);

  useEffect(() => {
    if (!connecting) return;
    const timer = window.setTimeout(() => setSlow(true), 4000);
    return () => window.clearTimeout(timer);
  }, [connecting]);

  async function connect(event: FormEvent) {
    event.preventDefault();
    setConnecting(true);
    setSlow(false);
    setError("");
    try {
      const result = await api<Discovered>(
        "/navidrome/discover",
        json("POST", { url: url.trim(), username: username.trim(), password, limit: 18 }),
      );
      await selectConnection(result.connection_id);
      setPreview({ tracks: result.tracks, version: result.server.version });
      setPassword("");
      setEditing(false);
      onConnected({ id: result.connection_id, url: result.server.url, username: username.trim() });
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Could not connect to the Navidrome server",
      );
    } finally {
      setConnecting(false);
    }
  }

  const host = (() => {
    try {
      return new URL(url).host;
    } catch {
      return url;
    }
  })();

  return (
    <StepFrame
      nav={nav}
      title={editing ? "Connect your music server" : "Your music server is connected"}
      description={
        editing
          ? "Echora reads your library from Navidrome and streams through it. It never changes your files."
          : "Echora will read your songs, lyrics and artwork from this server."
      }
      primary={
        editing ? (
          <Button form="setup-server" type="submit" loading={connecting}>
            {connecting ? "Connecting…" : "Connect"}
            {!connecting && <ArrowRight />}
          </Button>
        ) : (
          <Button onClick={nav.next}>
            Continue
            <ArrowRight />
          </Button>
        )
      }
      aside={
        editing && connection && !connecting ? (
          <Button
            variant="ghost"
            onClick={() => {
              setEditing(false);
              setError("");
            }}
          >
            Keep current server
          </Button>
        ) : null
      }
    >
      {editing ? (
        <form
          id="setup-server"
          onSubmit={connect}
          className="max-w-xl space-y-6"
          aria-busy={connecting}
        >
          <fieldset disabled={connecting} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="setup-server-url">Server URL</Label>
              <Input
                id="setup-server-url"
                required
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
              />
              <p className="text-[13px] text-muted-foreground">
                The address the Echora server uses to reach Navidrome. When both run on this machine
                in Docker, that is usually {DEFAULT_URL}.
              </p>
            </div>
            <div className="grid gap-6 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="setup-server-username">Username</Label>
                <Input
                  id="setup-server-username"
                  required
                  autoComplete="username"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="setup-server-password">Password</Label>
                <Input
                  id="setup-server-password"
                  required
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              </div>
            </div>
          </fieldset>
          {connecting && (
            <div role="status" className="border border-border bg-surface/60 p-4">
              <div className="flex items-center gap-2 text-[13px] font-medium">
                <Server className="size-4 text-primary" />
                Connecting to {host}…
              </div>
              <div className="mt-3 h-1 overflow-hidden bg-border">
                <div className="h-full w-1/3 animate-[setup-indeterminate_1.2s_ease-in-out_infinite] bg-primary" />
              </div>
              <p className="mt-3 text-[13px] text-muted-foreground">
                {slow
                  ? "Still working. A large library or a slow network can take a little longer."
                  : "Signing in and reading a few songs from your library."}
              </p>
            </div>
          )}
          {error && (
            <Notice tone="error" title="Connection failed">
              {error}. Check the address is reachable from the Echora server, and the username and
              password are right.
            </Notice>
          )}
        </form>
      ) : (
        connection && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center gap-4 border border-border bg-surface/60 p-4">
              <CircleCheck className="size-5 shrink-0 text-primary" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold">{connection.url}</p>
                <p className="text-[13px] text-muted-foreground">
                  Signed in as {connection.username}
                  {preview?.version ? ` · Navidrome ${preview.version}` : ""}
                </p>
              </div>
              <Button variant="outline" onClick={() => setEditing(true)}>
                Use a different server
              </Button>
            </div>
            {preview && preview.tracks.length > 0 && (
              <section aria-label="A few songs from your library">
                <p className="mb-3 text-[13px] text-muted-foreground">
                  A few songs from your library
                </p>
                <ul className="grid grid-cols-3 gap-3 sm:grid-cols-6">
                  {preview.tracks.slice(0, 18).map((track) => (
                    <li key={track.id} className="motion-fade min-w-0">
                      <div className="relative aspect-square overflow-hidden bg-raised">
                        {track.cover_art && (
                          <LoadingImage
                            sizes="120px"
                            src={coverArtUrl(connection.id, track.cover_art, 240)}
                            alt=""
                          />
                        )}
                      </div>
                      <p className="mt-1.5 truncate text-[12px] font-medium">{track.title}</p>
                      <p className="truncate text-[12px] text-muted-foreground">
                        {track.artist || "Unknown artist"}
                      </p>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        )
      )}
    </StepFrame>
  );
}
