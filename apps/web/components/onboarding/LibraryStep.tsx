"use client";

import { ArrowRight, Shuffle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { jobPresentation } from "../jobs/durableJobs";
import { useDurableJob } from "../jobs/useDurableJob";
import LoadingImage from "../media/LoadingImage";
import { coverArtUrl } from "../media/coverArt";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Notice } from "../ui/notice";
import { LoadingState } from "../ui/spinner";
import { cn } from "@/lib/utils";
import { api, json, type Connection, type Track } from "./onboardingApi";
import { StepFrame, type StepNav } from "./StepFrame";

type Scope = "library" | "sample";
type Compute = "local" | "modal";
type Status = { total: number; processed: number; missing: number };

const SAMPLE_SIZE = 40;
// A first taste: quick to analyze, and enough to try search and karaoke.
const PRESELECTED = 12;
const formatDuration = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`;

function RadioCard({
  checked,
  onSelect,
  title,
  detail,
  badge,
}: {
  checked: boolean;
  onSelect: () => void;
  title: string;
  detail: string;
  badge?: string;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={checked}
      onClick={onSelect}
      className={cn(
        "flex items-start gap-3 border p-4 text-left transition-colors",
        checked
          ? "border-primary/70 bg-primary/[.07]"
          : "border-border bg-surface/60 hover:border-border-strong hover:bg-surface",
      )}
    >
      <span
        className={cn(
          "mt-0.5 grid size-4 shrink-0 place-items-center rounded-full border",
          checked ? "border-primary" : "border-[#555]",
        )}
      >
        {checked && <span className="size-2 rounded-full bg-primary" />}
      </span>
      <span>
        <strong className="flex items-center gap-2 text-sm font-semibold">
          {title}
          {badge && (
            <span className="border border-primary/40 px-1.5 text-[11px] font-medium text-primary">
              {badge}
            </span>
          )}
        </strong>
        <span className="mt-1 block text-[13px] leading-relaxed text-muted-foreground">
          {detail}
        </span>
      </span>
    </button>
  );
}

/** Choose what to analyze first, start it, and follow it; Echora can be entered at any point. */
export default function LibraryStep({
  nav,
  connection,
  finish,
}: {
  nav: StepNav;
  connection: Connection;
  finish: () => Promise<void>;
}) {
  const { job, terminal, loading: jobLoading, track, dismiss } = useDurableJob(connection.id);
  const [scope, setScope] = useState<Scope>("library");
  const [status, setStatus] = useState<Status | null>(null);
  const [statusError, setStatusError] = useState("");
  const [modal, setModal] = useState<{ available: boolean; compute: Compute }>({
    available: false,
    compute: "local",
  });
  const [sample, setSample] = useState<Track[] | null>(null);
  const [sampleGeneration, setSampleGeneration] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [starting, setStarting] = useState(false);
  const [finishing, setFinishing] = useState(false);

  useEffect(() => {
    let active = true;
    api<Status>(`/navidrome/connections/${connection.id}/sync/status`)
      .then((value) => active && setStatus(value))
      .catch((reason) => active && setStatusError(reason.message));
    api<{ available: boolean; default: Compute }>("/settings/external-processing/availability")
      .then(
        (value) =>
          active &&
          setModal({
            available: value.available,
            compute: value.available ? value.default : "local",
          }),
      )
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [connection.id]);

  useEffect(() => {
    if (scope !== "sample") return;
    let active = true;
    api<{ tracks: Track[] }>(`/navidrome/connections/${connection.id}/sample?limit=${SAMPLE_SIZE}`)
      .then((value) => {
        if (!active) return;
        setSample(value.tracks);
        setSelected(new Set(value.tracks.slice(0, PRESELECTED).map((item) => item.id)));
      })
      .catch((reason) => {
        if (!active) return;
        setSample([]);
        toast.error("Could not load songs", { description: reason.message });
      });
    return () => {
      active = false;
    };
  }, [scope, connection.id, sampleGeneration]);

  const chosen = useMemo(
    () => (sample || []).filter((item) => selected.has(item.id)),
    [sample, selected],
  );
  const count = scope === "library" ? (status?.missing ?? 0) : chosen.length;

  async function start() {
    setStarting(true);
    try {
      const result =
        scope === "library"
          ? await api<{ job_id: string }>(
              `/navidrome/connections/${connection.id}/sync`,
              json("POST", { mode: "missing", verify_audio_hashes: true, compute: modal.compute }),
            )
          : await api<{ job_id: string }>(
              `/navidrome/connections/${connection.id}/import`,
              json("POST", { track_ids: chosen.map((item) => item.id) }),
            );
      track(result.job_id);
    } catch (reason) {
      toast.error("Could not start analysis", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setStarting(false);
    }
  }

  async function enter() {
    setFinishing(true);
    try {
      await finish();
    } finally {
      setFinishing(false);
    }
  }

  if (jobLoading)
    return (
      <StepFrame nav={nav} title="Analyze your music">
        <LoadingState label="Checking for analysis in progress…" />
      </StepFrame>
    );

  if (job) {
    const presentation = jobPresentation(job);
    const partial = job.status === "partial";
    const done = job.status === "complete" || partial;
    // Before the worker reports a total there is nothing to count yet.
    const warmingUp = !terminal && !job.total;
    // Any other finished state means the run stopped before analyzing anything usable.
    const failed = terminal && !done;
    return (
      <StepFrame
        nav={{ ...nav, back: null }}
        title={
          done ? "Your library is ready" : failed ? "Analysis stopped" : "Analyzing your music"
        }
        description={
          partial
            ? "Most songs were analyzed. A few could not be; Sync can retry them later."
            : done
              ? "Everything you picked has been analyzed."
              : failed
                ? "You can start again, or go to Echora and try later from Sync."
                : "This keeps running in the background. Go ahead and start listening; you can follow progress under Sync."
        }
        primary={
          <Button onClick={enter} loading={finishing}>
            {done ? "Enter Echora" : "Go to Echora"}
            <ArrowRight />
          </Button>
        }
        aside={
          terminal ? (
            <Button variant="ghost" onClick={() => void dismiss()}>
              {failed ? "Try again" : "Analyze more"}
            </Button>
          ) : null
        }
      >
        <div aria-live="polite" className="max-w-2xl space-y-6">
          <div className="border border-border bg-surface/60 p-5">
            <div className="flex items-baseline justify-between gap-4">
              <p className="text-sm font-semibold">
                {warmingUp ? "Getting ready…" : presentation.message}
              </p>
              {presentation.showPercent && !warmingUp && (
                <p className="text-sm tabular-nums text-muted-foreground">
                  {presentation.percent}%
                </p>
              )}
            </div>
            <div className="mt-3 h-1.5 overflow-hidden bg-border">
              {(presentation.showPercent && !warmingUp) || done ? (
                <div
                  className="h-full bg-primary transition-[width] duration-500"
                  style={{ width: `${done ? 100 : presentation.percent}%` }}
                />
              ) : (
                !terminal && (
                  <div className="h-full w-1/3 animate-[setup-indeterminate_1.2s_ease-in-out_infinite] bg-primary" />
                )
              )}
            </div>
            <p className="mt-3 text-[13px] text-muted-foreground">
              {warmingUp
                ? "Waiting for the analysis worker to pick this up."
                : job.track
                  ? `${job.track.artist || "Unknown artist"} — ${job.track.title}`
                  : presentation.detail}
            </p>
            {presentation.summary.length > 0 && (
              <ul className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
                {presentation.summary.map((item) => (
                  <li key={item} className="flex items-center gap-1.5">
                    <span aria-hidden="true" className="size-1 rounded-full bg-muted-foreground" />
                    {item}
                  </li>
                ))}
              </ul>
            )}
          </div>
          {job.error && <Notice tone="error">{job.error}</Notice>}
        </div>
      </StepFrame>
    );
  }

  const canStart =
    !starting && (scope === "library" ? Boolean(status && status.missing > 0) : chosen.length > 0);
  return (
    <StepFrame
      nav={nav}
      title="Analyze your music"
      description="Echora listens to each song once: sound, lyrics, karaoke timing and the features you switched on. You can keep listening while it works."
      primary={
        <Button onClick={start} loading={starting} disabled={!canStart}>
          {count > 0
            ? `Analyze ${count.toLocaleString()} ${count === 1 ? "song" : "songs"}`
            : "Analyze"}
          <ArrowRight />
        </Button>
      }
      aside={
        <Button variant="ghost" onClick={enter} loading={finishing}>
          <span className="md:hidden">Later</span>
          <span className="max-md:hidden">Skip, I&apos;ll do it later</span>
        </Button>
      }
    >
      <div className="space-y-8">
        <div role="radiogroup" aria-label="What to analyze" className="grid gap-3 sm:grid-cols-2">
          <RadioCard
            checked={scope === "library"}
            onSelect={() => setScope("library")}
            title="Entire library"
            badge="Recommended"
            detail={
              status
                ? status.missing === 0
                  ? `All ${status.total.toLocaleString()} songs are already analyzed.`
                  : `${status.missing.toLocaleString()} of ${status.total.toLocaleString()} songs still need analysis. Large libraries take a while; new songs are picked up by later syncs.`
                : statusError
                  ? "Could not count the songs on your server."
                  : "Counting the songs on your server…"
            }
          />
          <RadioCard
            checked={scope === "sample"}
            onSelect={() => setScope("sample")}
            title="Start with a few songs"
            detail="Try Echora on a random handful first, then sync the rest from Sync whenever you like."
          />
        </div>

        {scope === "library" && modal.available && (
          <section>
            <h2 className="mb-3 text-sm font-semibold">Where to run</h2>
            <div
              role="radiogroup"
              aria-label="Processing location"
              className="grid gap-3 sm:grid-cols-2"
            >
              <RadioCard
                checked={modal.compute === "local"}
                onSelect={() => setModal({ ...modal, compute: "local" })}
                title="This server"
                detail="Use this server's CPU or GPU."
              />
              <RadioCard
                checked={modal.compute === "modal"}
                onSelect={() => setModal({ ...modal, compute: "modal" })}
                title="Modal"
                detail="Cloud GPUs in your Modal workspace. GPU time is billed by Modal."
              />
            </div>
          </section>
        )}

        {scope === "sample" && (
          <section aria-label="Songs to analyze">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <p className="text-[13px] text-muted-foreground">
                {sample ? `${chosen.length} of ${sample.length} selected` : "Picking songs…"}
              </p>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!sample?.length}
                  onClick={() =>
                    setSelected(
                      selected.size === sample?.length
                        ? new Set()
                        : new Set((sample || []).map((item) => item.id)),
                    )
                  }
                >
                  {selected.size === sample?.length ? "Clear selection" : "Select all"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSample(null);
                    setSampleGeneration((value) => value + 1);
                  }}
                >
                  <Shuffle />
                  Other songs
                </Button>
              </div>
            </div>
            {!sample ? (
              <LoadingState label="Picking songs…" />
            ) : (
              <ul className="divide-y divide-border border border-border">
                {sample.map((item) => (
                  <li key={item.id}>
                    <label className="flex cursor-pointer items-center gap-3 px-3 py-2 hover:bg-hover">
                      <Checkbox
                        checked={selected.has(item.id)}
                        onCheckedChange={() =>
                          setSelected((current) => {
                            const next = new Set(current);
                            if (next.has(item.id)) next.delete(item.id);
                            else next.add(item.id);
                            return next;
                          })
                        }
                        aria-label={`Select ${item.title}`}
                      />
                      <span className="relative size-9 shrink-0 overflow-hidden bg-raised">
                        {item.cover_art && (
                          <LoadingImage
                            sizes="36px"
                            src={coverArtUrl(connection.id, item.cover_art, 84)}
                            alt=""
                          />
                        )}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[13px] font-medium">{item.title}</span>
                        <span className="block truncate text-[12px] text-muted-foreground">
                          {item.artist || "Unknown artist"}
                          {item.album ? ` · ${item.album}` : ""}
                        </span>
                      </span>
                      <span className="text-[12px] tabular-nums text-muted-foreground">
                        {formatDuration(item.duration)}
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}
      </div>
    </StepFrame>
  );
}
