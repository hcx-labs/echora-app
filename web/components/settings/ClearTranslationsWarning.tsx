"use client";

import { SettingsNotice } from "./SettingsNotice";
import { Button } from "../ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "../ui/dialog";

type Props = { busy: boolean; error?: string; onClose: () => void; onConfirm: () => void };
export default function ClearTranslationsWarning({ busy, error, onClose, onConfirm }: Props) {
  return <Dialog open onOpenChange={open => { if (!open && !busy) onClose(); }}><DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto" showCloseButton={!busy} aria-busy={busy} onEscapeKeyDown={event => { if (busy) event.preventDefault(); }} onInteractOutside={event => { if (busy) event.preventDefault(); }}><DialogHeader className="pr-6 text-left"><DialogTitle>Clear all translations?</DialogTitle><DialogDescription>This deletes stored translations for every user. Original lyrics and karaoke timing stay unchanged. This deletion cannot be undone.</DialogDescription></DialogHeader><SettingsNotice tone="warning" title="Before regenerating translations">Save your new prompt before the next Entire library sync.</SettingsNotice>{error && <SettingsNotice tone="error" title="Could not clear translations">{error}</SettingsNotice>}<DialogFooter><Button type="button" variant="outline" disabled={busy} onClick={onClose}>Keep translations</Button><Button type="button" variant="destructive" loading={busy} onClick={onConfirm}>{busy ? "Clearing…" : "Clear all translations"}</Button></DialogFooter></DialogContent></Dialog>;
}
