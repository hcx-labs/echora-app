"use client";

import { useState } from "react";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Label } from "../ui/label";
import { Notice } from "../ui/notice";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../ui/dialog";

/** Pipelines a full sync can bring up to date for every song of an older revision. */
export type ResyncPipeline = "karaoke" | "karaoke_stems" | "translation" | "themes";
const PIPELINES: { key: ResyncPipeline; label: string; description: string }[] = [
  {
    key: "karaoke",
    label: "Karaoke alignment",
    description: "Realign songs aligned by an older pipeline or model.",
  },
  {
    key: "karaoke_stems",
    label: "Karaoke stems",
    description: "Separate and encode again the songs whose stems use an older recipe.",
  },
  {
    key: "translation",
    label: "Translations",
    description:
      "Translate again the songs translated with another model or prompt; provider charges apply.",
  },
  {
    key: "themes",
    label: "Lyric themes",
    description: "Describe again the songs described with an older prompt; provider charges apply.",
  },
];

type Props = {
  onClose: () => void;
  onConfirm: (verifyAudioHashes: boolean, resync: ResyncPipeline[]) => void;
};
export default function FullSyncWarning({ onClose, onConfirm }: Props) {
  const [verifyAudioHashes, setVerifyAudioHashes] = useState(true);
  const [resync, setResync] = useState<ResyncPipeline[]>([]);
  const toggle = (key: ResyncPipeline, on: boolean) =>
    setResync((current) => (on ? [...current, key] : current.filter((item) => item !== key)));
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90dvh] overflow-auto">
        <DialogHeader>
          <DialogTitle>Sync the entire library?</DialogTitle>
          <DialogDescription>
            Refresh metadata and lyrics, fill missing analysis, and generate missing translations
            for your configured language pairs.
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center gap-3">
          <Checkbox
            id="verify-audio-hashes"
            checked={verifyAudioHashes}
            onCheckedChange={(checked) => setVerifyAudioHashes(checked === true)}
          />
          <Label htmlFor="verify-audio-hashes">Verify audio hashes</Label>
        </div>
        <p className="text-[13px] leading-relaxed text-muted-foreground">
          {verifyAudioHashes
            ? "Download source audio to verify track identity. This uses more bandwidth and time."
            : "Reuse known track identities without downloading audio just for hash comparison. Audio replaced under the same Navidrome ID may go undetected. New tracks and missing audio analysis can still require downloads."}
        </p>
        <fieldset className="grid gap-3 border-t border-border pt-4">
          <legend className="mb-1 text-[13px] font-medium text-foreground">
            Bring every song up to date
          </legend>
          <p className="-mt-1 text-[12px] leading-relaxed text-muted-foreground">
            Redoes only songs made by an older revision; up-to-date songs are left alone. To redo a
            single song regardless, use Processing information in its menu.
          </p>
          {PIPELINES.map((pipeline) => (
            <div key={pipeline.key} className="flex items-start gap-3">
              <Checkbox
                id={`resync-${pipeline.key}`}
                className="mt-0.5"
                checked={resync.includes(pipeline.key)}
                onCheckedChange={(checked) => toggle(pipeline.key, checked === true)}
              />
              <Label htmlFor={`resync-${pipeline.key}`} className="grid gap-0.5">
                <span>{pipeline.label}</span>
                <span className="text-[12px] font-normal text-muted-foreground">
                  {pipeline.description}
                </span>
              </Label>
            </div>
          ))}
        </fieldset>
        <Notice tone="warning" title="Provider charges may apply">
          Translations use your External AI endpoint and configured language pairs. Existing
          translations and analysis are reused unless chosen above.
        </Notice>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Go back
          </Button>
          <Button onClick={() => onConfirm(verifyAudioHashes, resync)}>
            Start entire-library sync
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
