"use client";

import { RotateCcw, X } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../ui/dialog";
import { LanguagePicker, languageName } from "../ui/language-picker";
import { Notice } from "../ui/notice";
import { LoadingState } from "../ui/spinner";
import styles from "./ProcessingDialog.module.css";

type Translation = {
  source: string;
  target: string;
  status: string;
  revision: string;
  updated_at?: string | null;
};
type Pipeline = {
  key: string;
  label: string;
  status: string;
  stored_revision?: string | null;
  current_revision?: string | null;
  detail?: string | null;
  resyncable: boolean;
  requested?: { options: Record<string, string>; requested_at: string } | null;
  updated_at?: string | null;
  language?: string | null;
  translations?: Translation[] | null;
};

const STATUS: Record<
  string,
  {
    label: string;
    variant: "success" | "warning" | "secondary" | "destructive" | "info" | "outline";
  }
> = {
  current: { label: "Up to date", variant: "success" },
  outdated: { label: "Out of date", variant: "warning" },
  missing: { label: "Not made", variant: "secondary" },
  failed: { label: "Failed", variant: "destructive" },
  stale: { label: "Lyrics changed", variant: "warning" },
  declined: { label: "Declined", variant: "secondary" },
  pending: { label: "Pending", variant: "info" },
  not_applicable: { label: "Not used", variant: "outline" },
};

function StatusBadge({ status }: { status: string }) {
  const known = STATUS[status] ?? { label: status, variant: "outline" as const };
  return <Badge variant={known.variant}>{known.label}</Badge>;
}

function Revision({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return (
    <div className={styles.revision}>
      <dt>{label}</dt>
      <dd title={value}>{value}</dd>
    </div>
  );
}

const path = (trackId: string, pipeline = "") =>
  `/api/v1/library/tracks/${encodeURIComponent(trackId)}/processing${pipeline ? `/${pipeline}` : ""}`;

/**
 * Each pipeline's revision for one song, whether it is out of date, and resyncs: a marked
 * pipeline is redone by the next sync, partial or full. Nothing runs from here.
 */
export default function ProcessingDialog({
  track,
  open,
  onOpenChange,
  onCloseAutoFocus,
}: {
  track: { id: string; title: string; artist?: string };
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCloseAutoFocus?: () => void;
}) {
  const [pipelines, setPipelines] = useState<Pipeline[] | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [language, setLanguage] = useState("");

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    fetch(path(track.id), { signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Could not load processing information");
        setError("");
        setPipelines(body.pipelines);
        const transcription = (body.pipelines as Pipeline[]).find(
          (item) => item.key === "transcription",
        );
        setLanguage(transcription?.requested?.options.language || transcription?.language || "");
      })
      .catch((reason) => {
        if (reason.name !== "AbortError")
          setError(
            reason instanceof Error ? reason.message : "Could not load processing information",
          );
      });
    return () => controller.abort();
  }, [open, track.id]);

  async function change(pipeline: Pipeline, request: boolean) {
    setBusy(pipeline.key);
    try {
      const response = await fetch(path(track.id, pipeline.key), {
        method: request ? "PUT" : "DELETE",
        headers: request ? { "content-type": "application/json" } : undefined,
        body: request
          ? JSON.stringify(pipeline.key === "transcription" ? { language } : {})
          : undefined,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Could not update the resync");
      setPipelines((current) =>
        (current ?? []).map((item) =>
          item.key === pipeline.key
            ? {
                ...item,
                requested: request
                  ? {
                      options: pipeline.key === "transcription" ? { language } : {},
                      requested_at: new Date().toISOString(),
                    }
                  : null,
              }
            : item,
        ),
      );
      toast.success(request ? `${pipeline.label}: marked for resync` : "Resync withdrawn", {
        description: request ? "The next sync, partial or full, redoes it." : track.title,
      });
    } catch (reason) {
      toast.error("Could not update the resync", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setBusy("");
    }
  }

  return (
    <Dialog open={open} onOpenChange={(next) => !busy && onOpenChange(next)}>
      <DialogContent
        className={`${styles.dialog} sm:max-w-[720px]`}
        showCloseButton={!busy}
        onCloseAutoFocus={onCloseAutoFocus}
      >
        <DialogHeader>
          <DialogTitle>Processing information</DialogTitle>
          <DialogDescription>
            {track.title}
            {track.artist ? ` · ${track.artist}` : ""}
          </DialogDescription>
        </DialogHeader>
        {error && <Notice tone="error">{error}</Notice>}
        {!pipelines && !error && <LoadingState label="Loading processing information" />}
        {pipelines && (
          <ul className={styles.list}>
            {pipelines.map((pipeline) => {
              const requested = pipeline.requested;
              const transcription = pipeline.key === "transcription";
              return (
                <li key={pipeline.key} className={styles.item}>
                  <div className={styles.head}>
                    <span className={styles.label}>{pipeline.label}</span>
                    <StatusBadge status={pipeline.status} />
                    {requested && (
                      <Badge variant="info">
                        Resync on next sync
                        {requested.options.language
                          ? ` · ${languageName(requested.options.language)}`
                          : ""}
                      </Badge>
                    )}
                  </div>
                  {pipeline.detail && <p className={styles.detail}>{pipeline.detail}</p>}
                  <dl className={styles.revisions}>
                    <Revision label="Stored" value={pipeline.stored_revision} />
                    {pipeline.status === "outdated" || !pipeline.stored_revision ? (
                      <Revision label="Current" value={pipeline.current_revision} />
                    ) : null}
                  </dl>
                  {pipeline.translations && pipeline.translations.length > 0 && (
                    <ul className={styles.translations}>
                      {pipeline.translations.map((item) => (
                        <li key={`${item.source}:${item.target}`}>
                          <span>
                            {item.source === "mul" ? "Foreign lines" : languageName(item.source)} →{" "}
                            {languageName(item.target)}
                          </span>
                          <StatusBadge status={item.status} />
                        </li>
                      ))}
                    </ul>
                  )}
                  {pipeline.resyncable && (
                    <div className={styles.actions}>
                      {transcription && !requested && (
                        <LanguagePicker
                          aria-label="Transcription language"
                          placeholder="Language to transcribe in"
                          value={language}
                          onChange={setLanguage}
                          className={styles.language}
                        />
                      )}
                      {requested ? (
                        <Button
                          variant="outline"
                          size="sm"
                          loading={busy === pipeline.key}
                          disabled={Boolean(busy)}
                          onClick={() => void change(pipeline, false)}
                        >
                          <X />
                          Withdraw resync
                        </Button>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          loading={busy === pipeline.key}
                          disabled={Boolean(busy) || (transcription && !language)}
                          onClick={() => void change(pipeline, true)}
                        >
                          <RotateCcw />
                          {transcription ? "Transcribe on next sync" : "Resync on next sync"}
                        </Button>
                      )}
                    </div>
                  )}
                  {transcription && pipeline.resyncable && !requested && (
                    <p className={styles.hint}>
                      Replaces the current lyrics with an AI transcription in this language when the
                      sync reaches the song. AI lyrics generation must be on in Settings.
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        )}
        {pipelines && (
          <Notice tone="info">
            Marked pipelines are redone by the next sync, partial or full, whatever their revision.
            Nothing changes until then.
          </Notice>
        )}
      </DialogContent>
    </Dialog>
  );
}
