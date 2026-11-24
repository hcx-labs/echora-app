"use client";

import { useState } from "react";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Label } from "../ui/label";
import { Notice } from "../ui/notice";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "../ui/dialog";

type Props = { onClose: () => void; onConfirm: (verifyAudioHashes: boolean) => void };
export default function FullSyncWarning({ onClose, onConfirm }: Props) {
  const [verifyAudioHashes, setVerifyAudioHashes] = useState(true);
  return <Dialog open onOpenChange={open => { if (!open) onClose(); }}><DialogContent><DialogHeader><DialogTitle>Sync the entire library?</DialogTitle><DialogDescription>Refresh metadata and lyrics, fill missing analysis, and generate missing translations for your configured language pairs.</DialogDescription></DialogHeader><div className="flex items-center gap-3"><Checkbox id="verify-audio-hashes" checked={verifyAudioHashes} onCheckedChange={checked => setVerifyAudioHashes(checked === true)} /><Label htmlFor="verify-audio-hashes">Verify audio hashes</Label></div><p className="text-[13px] leading-relaxed text-muted-foreground">{verifyAudioHashes ? "Download source audio to verify track identity. This uses more bandwidth and time." : "Reuse known track identities without downloading audio just for hash comparison. Audio replaced under the same Navidrome ID may go undetected. New tracks and missing audio analysis can still require downloads."}</p><Notice tone="warning" title="Provider charges may apply">Translations use your External AI endpoint and configured language pairs. Existing translations and analysis are reused.</Notice><DialogFooter><Button variant="outline" onClick={onClose}>Go back</Button><Button onClick={() => onConfirm(verifyAudioHashes)}>Start entire-library sync</Button></DialogFooter></DialogContent></Dialog>;
}
