"use client";

import { toast } from "sonner";
import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { useDurableJob } from "../jobs/useDurableJob";
import { jobPresentation, jobRequest } from "../jobs/durableJobs";
import BatchStatusList from "../jobs/BatchStatusList";
import type { Batch } from "../jobs/batchStatus";
import { runtimeConfig } from "../runtime/runtimeConfig";
import { Button } from "../ui/button";
import { Notice } from "../ui/notice";
import { Table, TableHeader, TableHead, TableBody, TableRow, TableCell } from "../ui/table";
import { Pane } from "../layout/pane";
import { LoadingState } from "../ui/spinner";
import FullSyncWarning from "./FullSyncWarning";
import { SidePanel } from "../layout/side-panel";
import { SectionHeader } from "../layout/section-header";
import { EmptyState } from "../layout/empty-state";
import SyncExplanation from "./SyncExplanation";

type Track = { id: string; title: string; artist?: string; album?: string };
type Status = {
  server: string;
  total: number;
  processed: number;
  missing: number;
  tracks: Track[];
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/analysis${path}`, options);
  const body = await response.json().catch(() => null);
  if (!response.ok || !body)
    throw new Error(body?.detail || `The request failed (${response.status})`);
  return body as T;
}

export default function SyncLibrary() {
  const [connectionId, setConnectionId] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const {
    job,
    active,
    terminal,
    error: jobError,
    loading: jobLoading,
    track,
    dismiss,
    dismissing,
  } = useDurableJob(connectionId);
  const [mode, setMode] = useState<"all" | "missing">("missing");
  // Where the sync computes; offered only when Modal is available to this user.
  const [compute, setCompute] = useState<"local" | "modal">("local");
  const [modalAvailable, setModalAvailable] = useState(false);
  const [confirmFullSync, setConfirmFullSync] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [cancelledJobId, setCancelledJobId] = useState("");
  const [busy, setBusy] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  // Progress phases of the running batches, so the model list can show what is in use now.
  // Kept with their job, so a new sync never shows the previous sync's phases.
  const [batchPhases, setBatchPhases] = useState<{ jobId: string; phases: string[] }>({
    jobId: "",
    phases: [],
  });

  function scan(connection: string) {
    setBusy(true);
    setError("");
    api<Status>(`/navidrome/connections/${connection}/sync/status`)
      .then(setStatus)
      .catch((reason) => setError(reason.message))
      .finally(() => setBusy(false));
  }

  useEffect(() => {
    fetch("/analysis/auth/me")
      .then((response) => response.json())
      .then((user) => {
        const connection = user.navidrome_connection_id || "";
        setConnectionId(connection);
        if (connection) scan(connection);
        else {
          setError("No Navidrome connection is configured");
          setBusy(false);
        }
      })
      .catch(() => {
        setError("Could not load your connection");
        setBusy(false);
      });
    api<{ available: boolean; default: "local" | "modal" }>(
      "/settings/external-processing/availability",
    )
      .then((result) => {
        setModalAvailable(result.available);
        setCompute(result.available ? result.default : "local");
      })
      .catch(() => setModalAvailable(false));
  }, []);

  useEffect(() => {
    if (!terminal || !connectionId) return;
    const timer = window.setTimeout(() => scan(connectionId), 400);
    return () => window.clearTimeout(timer);
  }, [connectionId, terminal]);

  async function start(verifyAudioHashes = true) {
    if (!connectionId || busy) return;
    setBusy(true);
    setStarting(true);
    try {
      const result = await api<{ job_id: string; status: string; existing: boolean }>(
        `/navidrome/connections/${connectionId}/sync`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            mode,
            verify_audio_hashes: verifyAudioHashes,
            compute: modalAvailable ? compute : "local",
          }),
        },
      );
      track(result.job_id);
      if (result.existing)
        toast.info("A sync is already running", { description: "Showing its progress." });
    } catch (reason) {
      toast.error("Could not start synchronization", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setBusy(false);
      setStarting(false);
    }
  }

  const jobId = job?.job_id || job?.id || "";
  const cancellationRequested = Boolean(
    job?.cancel_requested || (jobId && cancelledJobId === jobId),
  );
  async function cancelSync() {
    if (!active || !jobId || cancelling || cancellationRequested) return;
    setCancelling(true);
    try {
      await api(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
      setCancelledJobId(jobId);
    } catch (reason) {
      toast.error("Could not cancel sync", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setCancelling(false);
    }
  }

  const indeterminatePhases = new Set([
    "queued",
    "scanning",
    "starting",
    "planning",
    "models",
    "modal",
  ]);
  const indeterminate = Boolean(
    active && (job?.unit === "models" || indeterminatePhases.has(job?.phase || "")),
  );
  const presentation = jobPresentation(job);
  const { percent } = presentation;
  const progressDetail =
    job?.unit === "models"
      ? `${job.completed} of ${job.total} models ready`
      : indeterminate
        ? "This setup step does not report track progress"
        : presentation.detail;
  const showBatches = active && job?.unit === "batches";
  const batchJobId = showBatches ? job.job_id || job.id || "" : "";
  useEffect(() => {
    if (!batchJobId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        // Running batches are listed first.
        const { batches } = await jobRequest<{ batches: Batch[] }>(
          `/jobs/${encodeURIComponent(batchJobId)}/batches?limit=10`,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const phases = batches
          .filter((batch) => batch.status === "running" && batch.phase)
          .map((batch) => batch.phase);
        setBatchPhases((current) =>
          current.jobId === batchJobId && current.phases.join() === phases.join()
            ? current
            : { jobId: batchJobId, phases },
        );
      } catch {
        if (controller.signal.aborted) return;
      }
      timer = setTimeout(poll, runtimeConfig().batch_poll_ms);
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [batchJobId]);
  // Jobs without batches (such as the semantic fusion build) report their phase directly.
  const activePhases = showBatches
    ? batchPhases.jobId === batchJobId
      ? batchPhases.phases
      : []
    : active && job?.phase
      ? [job.phase]
      : [];
  const showJobDetails = active && !showBatches;
  const pendingLabel = `${status?.missing ?? "—"} new ${status?.missing === 1 ? "track" : "tracks"}`;
  const modes = [
    {
      value: "missing" as const,
      title: "New tracks only",
      detail: `Analyze ${status?.missing?.toLocaleString() ?? "—"} tracks not yet in Echora.`,
    },
    {
      value: "all" as const,
      title: "Entire library",
      detail: "Refresh metadata and lyrics, and fill any missing analysis.",
    },
  ];
  const locations = [
    {
      value: "local" as const,
      title: "This server",
      detail:
        "Process with this server's CPU or GPU. Runs the features switched on for this server.",
    },
    {
      value: "modal" as const,
      title: "Modal",
      detail:
        "Process on cloud GPUs in your Modal workspace. Uploads audio and lyrics; GPU time is billed by Modal.",
    },
  ];
  return (
    <>
      <div className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)] md:grid-cols-[340px_minmax(0,1fr)]">
        <div className="hidden min-h-0 border-r border-border bg-rail md:block">
          <SidePanel title="About sync" subtitle="What happens when you sync">
            {<SyncExplanation activePhases={activePhases} />}
          </SidePanel>
        </div>
        <Pane
          className="bg-workspace"
          label="Sync"
          title={
            active
              ? job?.kind === "semantic_fusion_build"
                ? "Building semantic fusion"
                : "Processing library"
              : busy
                ? "Scanning Navidrome"
                : "Sync library"
          }
          subtitle="Reads Navidrome without changing it. Existing analysis is kept."
          actions={
            <Button
              variant="outline"
              onClick={() => connectionId && scan(connectionId)}
              disabled={busy || !!active}
            >
              <RefreshCw className={busy ? "animate-spin" : undefined} />
              Rescan
            </Button>
          }
        >
          <div className="grid gap-10 pt-1">
            <details className="border border-border bg-rail p-4 md:hidden">
              <summary className="cursor-pointer text-[13px] font-medium">
                About sync and analysis models
              </summary>
              <div className="mt-4">
                <SyncExplanation activePhases={activePhases} />
              </div>
            </details>
            <section aria-label="Library totals">
              <p className="mb-3 truncate text-xs text-muted-foreground">
                {status?.server || "Reading connection…"}
              </p>
              <dl className="grid grid-cols-3 border border-border bg-surface/60 max-sm:grid-cols-1">
                {(
                  [
                    ["In Navidrome", status?.total, false],
                    ["In Echora", status?.processed, false],
                    ["Not yet analyzed", status?.missing, true],
                  ] as const
                ).map(([label, value, highlight]) => (
                  <div
                    key={label}
                    className="border-border px-5 py-4 not-first:border-l max-sm:not-first:border-t max-sm:not-first:border-l-0"
                  >
                    <dt className="text-xs text-muted-foreground">{label}</dt>
                    <dd
                      className={`mt-1 text-[28px] leading-tight font-semibold tabular-nums ${highlight && value ? "text-primary" : ""}`}
                    >
                      {typeof value === "number" ? value.toLocaleString() : "—"}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
            {job ? (
              <section aria-label="Current sync job" className="border border-border bg-surface/60">
                <div className="flex flex-wrap items-start justify-between gap-4 p-5">
                  <div className="min-w-0 space-y-1">
                    <p className="text-xs font-medium tracking-wide text-primary uppercase">
                      {job.phase}
                    </p>
                    <p className="text-[15px] font-semibold">{presentation.message}</p>
                    <p className="text-[13px] text-muted-foreground">{progressDetail}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    {!terminal && !indeterminate && presentation.showPercent && (
                      <span className="text-2xl font-semibold tabular-nums">{percent}%</span>
                    )}
                    {active && (
                      <Button
                        variant="outline"
                        loading={cancelling}
                        disabled={cancellationRequested}
                        onClick={cancelSync}
                      >
                        {cancellationRequested
                          ? "Cancellation requested"
                          : cancelling
                            ? "Cancelling…"
                            : "Cancel sync"}
                      </Button>
                    )}
                    {terminal && (
                      <Button
                        variant="outline"
                        onClick={dismiss}
                        loading={dismissing}
                        aria-label="Dismiss sync results"
                      >
                        {dismissing ? "Dismissing…" : "Dismiss"}
                      </Button>
                    )}
                  </div>
                </div>
                {(indeterminate || presentation.showPercent) && (
                  <div
                    role="progressbar"
                    aria-label={indeterminate ? "Preparing analysis" : "Library sync progress"}
                    aria-valuemin={indeterminate ? undefined : 0}
                    aria-valuemax={indeterminate ? undefined : 100}
                    aria-valuenow={indeterminate ? undefined : percent}
                    className="h-1 overflow-hidden bg-raised"
                  >
                    <div
                      className={`h-full bg-primary transition-[width] ${indeterminate ? "animate-pulse" : ""}`}
                      style={{
                        width: indeterminate ? "100%" : `${percent}%`,
                        opacity: indeterminate ? 0.5 : 1,
                      }}
                    />
                  </div>
                )}
                {(job.error || (terminal && presentation.summary.length > 0)) && (
                  <div className="space-y-2 border-t border-border p-5">
                    {job.error && (
                      <p role="alert" className="text-[13px] text-destructive">
                        {job.error}
                      </p>
                    )}
                    {terminal && (
                      <ul className="space-y-1 text-[13px] text-muted-foreground">
                        {presentation.summary.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
                <p className="border-t border-border px-5 py-3 text-xs break-all text-subtle-foreground">
                  Job {jobId} · {job.status}
                </p>
              </section>
            ) : (
              <section aria-label="Sync controls">
                <SectionHeader title="What to sync" />
                <div role="radiogroup" aria-label="Sync mode" className="grid gap-3 sm:grid-cols-2">
                  {modes.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      role="radio"
                      aria-checked={mode === option.value}
                      onClick={() => setMode(option.value)}
                      className={`flex items-start gap-3 border p-4 text-left transition-colors ${mode === option.value ? "border-primary/70 bg-primary/[.07]" : "border-border bg-surface/60 hover:border-border-strong hover:bg-surface"}`}
                    >
                      <span
                        className={`mt-0.5 grid size-4 shrink-0 place-items-center rounded-full border ${mode === option.value ? "border-primary" : "border-[#555]"}`}
                      >
                        {mode === option.value && (
                          <span className="size-2 rounded-full bg-primary" />
                        )}
                      </span>
                      <span>
                        <strong className="block text-sm font-semibold">{option.title}</strong>
                        <span className="mt-1 block text-[13px] leading-relaxed text-muted-foreground">
                          {option.detail}
                        </span>
                      </span>
                    </button>
                  ))}
                </div>
                {modalAvailable && (
                  <>
                    <div className="mt-6">
                      <SectionHeader title="Where to run" />
                    </div>
                    <div
                      role="radiogroup"
                      aria-label="Processing location"
                      className="grid gap-3 sm:grid-cols-2"
                    >
                      {locations.map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          role="radio"
                          aria-checked={compute === option.value}
                          onClick={() => setCompute(option.value)}
                          className={`flex items-start gap-3 border p-4 text-left transition-colors ${compute === option.value ? "border-primary/70 bg-primary/[.07]" : "border-border bg-surface/60 hover:border-border-strong hover:bg-surface"}`}
                        >
                          <span
                            className={`mt-0.5 grid size-4 shrink-0 place-items-center rounded-full border ${compute === option.value ? "border-primary" : "border-[#555]"}`}
                          >
                            {compute === option.value && (
                              <span className="size-2 rounded-full bg-primary" />
                            )}
                          </span>
                          <span>
                            <strong className="block text-sm font-semibold">{option.title}</strong>
                            <span className="mt-1 block text-[13px] leading-relaxed text-muted-foreground">
                              {option.detail}
                            </span>
                          </span>
                        </button>
                      ))}
                    </div>
                  </>
                )}
                <div className="mt-4 flex justify-end">
                  <Button
                    size="lg"
                    loading={starting}
                    onClick={() => {
                      if (mode === "all") setConfirmFullSync(true);
                      else void start();
                    }}
                    disabled={
                      busy ||
                      jobLoading ||
                      !!jobError ||
                      !status ||
                      (mode === "missing" && status.missing === 0)
                    }
                  >
                    Start sync
                  </Button>
                </div>
              </section>
            )}
            {error && (
              <Notice tone="error" title="Could not load sync status">
                {error}
              </Notice>
            )}
            <section className="min-w-0">
              <SectionHeader
                title={
                  showBatches
                    ? "Sync batches"
                    : showJobDetails
                      ? "Processing details"
                      : "Waiting in Navidrome"
                }
                actions={!active && status?.tracks.length ? <span>{pendingLabel}</span> : undefined}
              />
              {showBatches ? (
                <BatchStatusList
                  key={job.job_id || job.id}
                  jobId={job.job_id || job.id!}
                  active={active}
                />
              ) : showJobDetails ? (
                <p className="text-[13px] text-muted-foreground">
                  {job?.kind === "semantic_fusion_build"
                    ? "Combining audio and lyrics embeddings across the Echora library. This step has no track batches."
                    : "Preparing sync batches. Track progress appears when processing begins."}
                </p>
              ) : status?.tracks.length ? (
                <div className="border border-border">
                  <Table className="table-fixed">
                    <TableHeader>
                      <TableRow className="hover:bg-transparent">
                        <TableHead>Track</TableHead>
                        <TableHead>Artist</TableHead>
                        <TableHead className="max-md:hidden">Album</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {status.tracks.map((track) => (
                        <TableRow key={track.id}>
                          <TableCell className="truncate font-medium">{track.title}</TableCell>
                          <TableCell className="truncate text-muted-foreground">
                            {track.artist || "Unknown artist"}
                          </TableCell>
                          <TableCell className="truncate text-muted-foreground max-md:hidden">
                            {track.album || "Unknown album"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : busy && !status ? (
                <LoadingState label="Scanning Navidrome for new tracks…" />
              ) : (
                <div role="status" className="border border-dashed border-border">
                  <EmptyState
                    title={
                      busy
                        ? "Scanning the catalog"
                        : error || jobError
                          ? "Catalog unavailable"
                          : "Everything is analyzed"
                    }
                    description={
                      busy
                        ? "Checking Navidrome for tracks that need analysis."
                        : error || jobError
                          ? "Resolve the connection error, then rescan your library."
                          : "Rescan after adding music, or sync the entire library to refresh metadata."
                    }
                  />
                </div>
              )}
            </section>
          </div>
        </Pane>
      </div>
      {confirmFullSync && (
        <FullSyncWarning
          onClose={() => setConfirmFullSync(false)}
          onConfirm={(verifyAudioHashes) => {
            setConfirmFullSync(false);
            void start(verifyAudioHashes);
          }}
        />
      )}
    </>
  );
}
