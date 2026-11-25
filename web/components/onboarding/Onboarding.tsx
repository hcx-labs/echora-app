"use client";

import { AudioLines, Check, Lock, LogOut } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import AnalysisFeatures from "../settings/AnalysisFeatures";
import AppearanceSettings from "../settings/AppearanceSettings";
import ExternalAISettings from "../settings/ExternalAISettings";
import ExternalProcessingSettings from "../settings/ExternalProcessingSettings";
import MotionArtworkSettings from "../settings/MotionArtworkSettings";
import AppLoading from "../shell/AppLoading";
import { cn } from "@/lib/utils";
import { Button } from "../ui/button";
import LibraryStep from "./LibraryStep";
import { api, json, selectConnection, type Connection, type SetupSettings } from "./onboardingApi";
import ServerStep from "./ServerStep";
import { StepFrame, type StepNav } from "./StepFrame";
import WelcomeStep from "./WelcomeStep";

type StepId = "welcome" | "server" | "processing" | "features" | "ai" | "appearance" | "library";
type Step = {
  id: StepId;
  title: string;
  detail: string;
  group: string;
  admin?: boolean;
  optional?: boolean;
  needsServer?: boolean;
};

const STEPS: Step[] = [
  { id: "welcome", title: "Welcome", detail: "Your name and timezone.", group: "Get started" },
  {
    id: "server",
    title: "Music server",
    detail: "Connect Navidrome so Echora can read your library.",
    group: "Get started",
  },
  {
    id: "processing",
    title: "Where to process",
    detail: "This server, or cloud GPUs on Modal.",
    group: "Server setup",
    admin: true,
    optional: true,
  },
  {
    id: "features",
    title: "Analysis features",
    detail: "Karaoke timing, AI lyrics, humming search, motion artwork.",
    group: "Server setup",
    admin: true,
  },
  {
    id: "ai",
    title: "External AI",
    detail: "Translations, refined transcripts and song themes.",
    group: "Server setup",
    admin: true,
    optional: true,
  },
  {
    id: "appearance",
    title: "Appearance",
    detail: "Karaoke style and the backdrop behind your music.",
    group: "Make it yours",
    optional: true,
  },
  {
    id: "library",
    title: "Analyze your music",
    detail: "All of it, or a few songs to start.",
    group: "Make it yours",
    needsServer: true,
  },
];

// Ticks are a convenience kept in this browser, per account, so another account (or a reset
// server) never inherits them. Whether a server is connected always comes from the server.
const progressKey = (email: string) => `echora:onboarding-done:${email}`;

function readDone(email: string): Set<StepId> {
  try {
    return new Set(JSON.parse(localStorage.getItem(progressKey(email)) || "[]"));
  } catch {
    return new Set();
  }
}

function stepFromPath(): string | undefined {
  return window.location.pathname.split("/")[2];
}

export default function Onboarding({ initialStep }: { initialStep?: string }) {
  const [settings, setSettings] = useState<SetupSettings | null>(null);
  const [connection, setConnection] = useState<Connection | null>(null);
  const [current, setCurrent] = useState<StepId>("welcome");
  const [done, setDone] = useState<Set<StepId>>(() => new Set());

  const steps = STEPS.filter((step) => !step.admin || settings?.profile.is_admin);

  // Signed-out visitors go to sign-in and finished accounts go home; everyone else resumes here.
  useEffect(() => {
    let active = true;
    (async () => {
      const response = await fetch("/api/v1/auth/me");
      if (response.status === 401) return window.location.replace("/login");
      if (!response.ok) throw new Error("Could not read your account");
      const user = await response.json();
      if (user.onboarding_complete) return window.location.replace("/home");
      const value = await api<SetupSettings>("/settings");
      if (!active) return;
      // The library follows the connection the user picked, which may differ from the latest one.
      const saved =
        value.navidrome &&
        (!user.navidrome_connection_id || value.navidrome.id === user.navidrome_connection_id)
          ? value.navidrome
          : null;
      const finished = readDone(value.profile.email);
      if (!saved) finished.delete("server");
      const visible = STEPS.filter((step) => !step.admin || value.profile.is_admin);
      const usable = (step: Step | undefined) => Boolean(step && (!step.needsServer || saved));
      const wanted = visible.find((step) => step.id === (initialStep || stepFromPath()));
      // Without a step in the address, resume at the first unfinished one.
      const resume = visible.find((step) => !finished.has(step.id) && usable(step)) ?? visible[0];
      setSettings(value);
      setConnection(saved);
      setDone(finished);
      setCurrent(usable(wanted) ? wanted!.id : resume.id);
      window.history.replaceState({}, "", `/setup/${usable(wanted) ? wanted!.id : resume.id}`);
    })().catch((reason) => {
      if (active) toast.error("Could not load setup", { description: reason.message });
    });
    return () => {
      active = false;
    };
  }, [initialStep]);

  useEffect(() => {
    const onHistory = () => {
      const wanted = steps.find((step) => step.id === stepFromPath());
      if (wanted && (!wanted.needsServer || connection)) setCurrent(wanted.id);
    };
    window.addEventListener("popstate", onHistory);
    return () => window.removeEventListener("popstate", onHistory);
  }, [steps, connection]);

  const go = useCallback((id: StepId) => {
    setCurrent(id);
    if (stepFromPath() !== id) window.history.pushState({}, "", `/setup/${id}`);
    document.getElementById("setup-main")?.scrollTo({ top: 0 });
  }, []);

  const email = settings?.profile.email ?? "";
  const markDone = useCallback(
    (id: StepId) => {
      setDone((previous) => {
        const next = new Set(previous).add(id);
        try {
          localStorage.setItem(progressKey(email), JSON.stringify([...next]));
        } catch {}
        return next;
      });
    },
    [email],
  );

  async function finish() {
    try {
      if (connection) await selectConnection(connection.id, true);
      else await api("/settings/onboarding", json("PUT", { complete: true, connection_id: null }));
      try {
        localStorage.removeItem(progressKey(email));
      } catch {}
      window.dispatchEvent(
        new CustomEvent("echora:user-update", { detail: { onboarding_complete: true } }),
      );
      window.location.assign(new URL("/home", window.location.origin));
    } catch (reason) {
      toast.error("Could not finish setup", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    }
  }

  async function signOut() {
    try {
      await fetch("/api/v1/auth/logout", { method: "POST" });
    } finally {
      window.location.assign(new URL("/login", window.location.origin));
    }
  }

  if (!settings) return <AppLoading />;

  const index = Math.max(
    0,
    steps.findIndex((step) => step.id === current),
  );
  const step = steps[index];
  const nav: StepNav = {
    index,
    total: steps.length,
    optional: Boolean(step.optional),
    back: index > 0 ? () => go(steps[index - 1].id) : null,
    next: () => {
      markDone(step.id);
      const following = steps[index + 1];
      if (!following) return;
      // The last step needs a server; send people to connect one rather than to a locked step.
      if (following.needsServer && !connection) {
        toast.info("Connect your music server first", {
          description: "Echora needs it to find the music to analyze.",
        });
        go("server");
      } else go(following.id);
    },
  };
  const locked = (item: Step) => Boolean(item.needsServer && !connection);
  const groups = Array.from(new Set(steps.map((item) => item.group)));
  const percent = Math.round((done.size / steps.length) * 100);

  return (
    <div className="grid h-dvh grid-rows-[auto_minmax(0,1fr)] bg-nav md:grid-cols-[300px_minmax(0,1fr)] md:grid-rows-[minmax(0,1fr)]">
      <aside className="hidden min-h-0 flex-col border-r border-border bg-rail md:flex">
        <div className="px-6 pt-7">
          <div className="flex items-center gap-2.5 text-xl font-extrabold tracking-tight">
            <AudioLines className="size-5 text-primary" />
            Echora
          </div>
          <p className="mt-6 text-[13px] font-medium">Setting up</p>
          <div className="mt-2 h-1 bg-border" aria-hidden="true">
            <div
              className="h-full bg-primary transition-[width] duration-500"
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
        <nav aria-label="Setup steps" className="min-h-0 flex-1 overflow-y-auto px-3 py-6">
          {groups.map((group) => (
            <div key={group} className="mb-5">
              <p className="mb-1.5 px-3 text-[11px] font-semibold tracking-[0.12em] text-subtle-foreground uppercase">
                {group}
              </p>
              <ol>
                {steps
                  .filter((item) => item.group === group)
                  .map((item) => {
                    const position = steps.indexOf(item);
                    const isCurrent = item.id === current;
                    const isDone =
                      done.has(item.id) &&
                      !isCurrent &&
                      (item.id !== "server" || Boolean(connection));
                    const isLocked = locked(item);
                    return (
                      <li key={item.id}>
                        <button
                          type="button"
                          disabled={isLocked}
                          aria-current={isCurrent ? "step" : undefined}
                          onClick={() => go(item.id)}
                          title={isLocked ? "Connect your music server first" : undefined}
                          className={cn(
                            "flex w-full items-start gap-3 px-3 py-2 text-left transition-colors",
                            isCurrent ? "bg-primary/[.08]" : "hover:bg-hover",
                            isLocked && "cursor-not-allowed opacity-50 hover:bg-transparent",
                          )}
                        >
                          <span
                            className={cn(
                              "mt-px grid size-6 shrink-0 place-items-center border text-xs tabular-nums",
                              isCurrent
                                ? "border-primary bg-primary text-primary-foreground"
                                : isDone
                                  ? "border-primary/60 text-primary"
                                  : "border-border-strong text-muted-foreground",
                            )}
                          >
                            {isDone ? (
                              <Check className="size-3.5" />
                            ) : isLocked ? (
                              <Lock className="size-3" />
                            ) : (
                              position + 1
                            )}
                          </span>
                          <span className="min-w-0">
                            <span
                              className={cn(
                                "block text-[13px] font-medium",
                                isCurrent ? "text-foreground" : "text-muted-foreground",
                              )}
                            >
                              {item.title}
                              {item.optional && (
                                <span className="font-normal text-subtle-foreground">
                                  {" "}
                                  · optional
                                </span>
                              )}
                            </span>
                            {isCurrent && (
                              <span className="mt-0.5 block text-[12px] leading-snug text-muted-foreground">
                                {item.detail}
                              </span>
                            )}
                          </span>
                        </button>
                      </li>
                    );
                  })}
              </ol>
            </div>
          ))}
        </nav>
        <div className="flex items-center gap-2 border-t border-border px-6 py-4">
          <p className="min-w-0 flex-1 truncate text-[12px] text-muted-foreground">
            {settings.profile.email}
          </p>
          <button
            type="button"
            onClick={signOut}
            className="flex items-center gap-1.5 text-[12px] text-muted-foreground hover:text-foreground"
          >
            <LogOut className="size-3.5" />
            Sign out
          </button>
        </div>
      </aside>

      <header className="border-b border-border bg-nav px-[var(--gutter)] py-3 md:hidden">
        <div className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 font-extrabold tracking-tight">
            <AudioLines className="size-4 text-primary" />
            Echora
          </span>
          <span className="text-[12px] text-muted-foreground tabular-nums">
            {index + 1} / {steps.length} · {step.title}
          </span>
        </div>
        <div className="mt-2.5 h-1 bg-border" aria-hidden="true">
          <div
            className="h-full bg-primary transition-[width] duration-500"
            style={{ width: `${((index + 1) / steps.length) * 100}%` }}
          />
        </div>
      </header>

      <main id="setup-main" className="min-h-0 overflow-y-auto bg-workspace">
        <div key={step.id} className="motion-fade flex min-h-full flex-col">
          {step.id === "welcome" && (
            <WelcomeStep
              nav={nav}
              settings={settings}
              plan={steps.map(({ title, detail }) => ({ title, detail }))}
              onSaved={(next) => setSettings({ ...settings, ...next })}
            />
          )}
          {step.id === "server" && (
            <ServerStep
              nav={nav}
              connection={connection}
              onConnected={(value) => {
                setConnection(value);
                markDone("server");
              }}
            />
          )}
          {step.id === "processing" && (
            <StepFrame
              nav={nav}
              title="Where should analysis run?"
              description="Echora analyzes on this server by default. If this machine has no GPU, or you want the first sync to go faster, it can send the work to cloud GPUs in your own Modal workspace instead. Leave this off to keep everything here."
            >
              <ExternalProcessingSettings />
            </StepFrame>
          )}
          {step.id === "features" && (
            <StepFrame
              nav={nav}
              title="What should Echora analyze?"
              description="Search, mixes and voice filters always run. The features below are optional; each can run on this server, on Modal, or both."
            >
              <div className="space-y-10">
                <AnalysisFeatures />
                <details className="group border border-border bg-surface/60">
                  <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold marker:hidden">
                    Motion artwork
                    <span className="ml-2 font-normal text-muted-foreground">
                      Looping animated covers for the full-screen player. Needs a large model
                      download.
                    </span>
                  </summary>
                  <div className="border-t border-border px-4 py-4">
                    <MotionArtworkSettings connectionId={connection?.id ?? null} />
                  </div>
                </details>
              </div>
            </StepFrame>
          )}
          {step.id === "ai" && (
            <StepFrame
              nav={nav}
              title="Connect an AI model"
              description="With an AI endpoint such as Claude or any OpenAI-compatible API, Echora can translate lyrics, clean up transcribed lyrics, and describe each song's themes so you can build mixes like “songs about leaving home”. Without one, everything else still works."
            >
              <ExternalAISettings />
            </StepFrame>
          )}
          {step.id === "appearance" && (
            <StepFrame
              nav={nav}
              title="Make it look right"
              description="How lyrics light up while a song plays, and what moves behind them. Saved in this browser; more options are in Settings → Appearance."
            >
              <AppearanceSettings essentials quiet />
            </StepFrame>
          )}
          {step.id === "library" &&
            (connection ? (
              <LibraryStep nav={nav} connection={connection} finish={finish} />
            ) : (
              <StepFrame
                nav={nav}
                title="Connect your music server first"
                description="Echora analyzes the music on your Navidrome server, so connect it before starting."
                primary={<Button onClick={() => go("server")}>Connect a server</Button>}
              >
                <span />
              </StepFrame>
            ))}
        </div>
      </main>
    </div>
  );
}
