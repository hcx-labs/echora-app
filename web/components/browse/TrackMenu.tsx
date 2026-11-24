"use client";

import { Ellipsis, Languages, ListPlus, Music2, Play, SquarePen, VolumeX } from "lucide-react";
import { useRef, useState } from "react";
import type { PlayerTrack } from "../player/PlayerProvider";
import { usePlayer } from "../player/PlayerProvider";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Notice } from "../ui/notice";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "../ui/dialog";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator } from "../ui/dropdown-menu";
import styles from "./TrackMenu.module.css";

type Track = { id: string; title: string; artist?: string; duration_seconds: number; source_id?: string; connectionId?: string; streamUrl?: string; coverUrl?: string; lyrics_status?: string; lyrics_text?: string; lyrics_language?: string };
type Lyrics = { text?: string | null; language?: string | null; availability_status?: string | null };

export default function TrackMenu({ track, onSaved }: { track: Track; onSaved: () => void }) {
  const player = usePlayer();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  const [transcriptionOpen, setTranscriptionOpen] = useState(false);
  const [transcriptionLanguage, setTranscriptionLanguage] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [language, setLanguage] = useState("");
  const [busy, setBusy] = useState(false);
  const closeTranscription = () => { if (!busy) setTranscriptionOpen(false); };
  const closeEditor = () => { if (!busy) setEditorOpen(false); };
  const playable = Boolean(track.streamUrl && track.source_id);
  const playerTrack: PlayerTrack | null = playable ? { id: track.id, title: track.title, artist: track.artist, durationSeconds: track.duration_seconds, streamUrl: track.streamUrl!, coverUrl: track.coverUrl, connectionId: track.connectionId, sourceId: track.source_id } : null;

  async function openEditor() {
    setOpen(false); setBusy(true);
    try {
      const response = await fetch(`/analysis/library/tracks/${encodeURIComponent(track.id)}/lyrics`);
      const body = await response.json() as Lyrics;
      if (!response.ok) throw new Error("Could not load lyrics");
      setLyrics(body.text || ""); setLanguage(body.language || ""); setEditorOpen(true);
    } catch (reason) { toast.error("Could not load lyrics", { description: reason instanceof Error && reason.message !== "Could not load lyrics" ? reason.message : undefined }); }
    finally { setBusy(false); }
  }

  async function saveLyrics() {
    setBusy(true);
    try {
      const response = await fetch(`/analysis/library/tracks/${encodeURIComponent(track.id)}/lyrics`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: lyrics, language: language || null }) });
      const body = await response.json(); if (!response.ok) throw new Error(body.detail || "Could not save lyrics");
      setEditorOpen(false); window.dispatchEvent(new CustomEvent("echora:lyrics-update", { detail: track.id })); onSaved(); toast.success("Lyrics saved", { description: track.title });
    } catch (reason) { toast.error("Could not save lyrics", { description: reason instanceof Error && reason.message !== "Could not save lyrics" ? reason.message : undefined }); }
    finally { setBusy(false); }
  }

  async function forceTranscription() {
    setBusy(true);
    try {
      const response = await fetch(`/analysis/library/tracks/${encodeURIComponent(track.id)}/lyrics/transcription-language`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ language: transcriptionLanguage }) });
      const body = await response.json(); if (!response.ok) throw new Error(body.detail || "Could not queue transcription");
      setTranscriptionOpen(false); window.dispatchEvent(new CustomEvent("echora:lyrics-update", { detail: track.id })); onSaved(); toast.success("Transcription queued", { description: "It runs during the next sync." });
    } catch (reason) { toast.error("Could not queue transcription", { description: reason instanceof Error && reason.message !== "Could not queue transcription" ? reason.message : undefined }); }
    finally { setBusy(false); }
  }

  async function mark(status: "instrumental" | "missing") {
    setOpen(false); setBusy(true);
    try {
      const response = await fetch(`/analysis/library/tracks/${encodeURIComponent(track.id)}/lyrics/status`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ status }) });
      const body = await response.json(); if (!response.ok) throw new Error(body.detail || "Could not update lyric status");
      window.dispatchEvent(new CustomEvent("echora:lyrics-update", { detail: track.id })); onSaved(); toast.success(status === "instrumental" ? "Marked as instrumental" : "No longer marked as instrumental", { description: track.title });
    } catch (reason) { toast.error("Could not update lyric status", { description: reason instanceof Error && reason.message !== "Could not update lyric status" ? reason.message : undefined }); }
    finally { setBusy(false); }
  }

  return <div className={styles.root} onClick={event => event.stopPropagation()}>
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild><Button ref={triggerRef} type="button" variant="ghost" size="icon-sm" aria-label={`Actions for ${track.title}`} className="text-subtle-foreground hover:text-foreground"><Ellipsis /></Button></DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuItem disabled={!playerTrack} onSelect={() => { if (playerTrack) player.play(playerTrack); }}><Play />Play now</DropdownMenuItem>
        <DropdownMenuItem disabled={!playerTrack} onSelect={() => { if (playerTrack) player.playNext(playerTrack); }}><ListPlus />Play next</DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem disabled={busy} onSelect={() => void openEditor()}><SquarePen />Edit lyrics</DropdownMenuItem>
        <DropdownMenuItem disabled={busy} onSelect={() => setTranscriptionOpen(true)}><Languages />Transcribe in language</DropdownMenuItem>
        <DropdownMenuItem disabled={busy} onSelect={() => void mark(track.lyrics_status === "instrumental" ? "missing" : "instrumental")}>{track.lyrics_status === "instrumental" ? <Music2 /> : <VolumeX />}{track.lyrics_status === "instrumental" ? "Mark non-instrumental" : "Mark instrumental"}</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
    <Dialog open={transcriptionOpen} onOpenChange={next => { if (!busy) setTranscriptionOpen(next); }}>
      <DialogContent className={`${styles.dialog} sm:max-w-[620px]`} showCloseButton={!busy} onCloseAutoFocus={() => triggerRef.current?.focus()}>
        <DialogHeader><DialogTitle>Transcribe {track.title}</DialogTitle><DialogDescription>{track.artist || "Choose the recording language."}</DialogDescription></DialogHeader>
        <label>Language code<Input value={transcriptionLanguage} onChange={event => setTranscriptionLanguage(event.target.value)} placeholder="en or pt-BR" maxLength={5} /></label>
        <Notice tone="warning" title="Replaces the current lyrics">MOSS transcribes in this language during the next sync. AI lyrics generation must be enabled in Settings.</Notice>
        <DialogFooter><Button variant="outline" onClick={closeTranscription} disabled={busy}>Cancel</Button><Button onClick={() => void forceTranscription()} loading={busy} disabled={ !/^[a-z]{2,3}(?:-[A-Z]{2})?$/.test(transcriptionLanguage)}>{busy ? "Queueing" : "Queue transcription"}</Button></DialogFooter>
      </DialogContent>
    </Dialog>
    <Dialog open={editorOpen} onOpenChange={next => { if (!busy) setEditorOpen(next); }}>
      <DialogContent className={`${styles.dialog} sm:max-w-[620px]`} showCloseButton={!busy} onCloseAutoFocus={() => triggerRef.current?.focus()}>
        <DialogHeader><DialogTitle>Lyrics for {track.title}</DialogTitle><DialogDescription>{track.artist || "Edit the lyrics and optional language."}</DialogDescription></DialogHeader>
        <label>Language, optional<Input value={language} onChange={event => setLanguage(event.target.value)} placeholder="en" maxLength={12} /></label>
        <label>Lyrics<Textarea className={styles.lyrics} value={lyrics} onChange={event => setLyrics(event.target.value)} placeholder="Paste or write the lyrics" /></label>
        <Notice tone="info">Saved lyrics skip AI transcription. Karaoke timing is regenerated during the next sync.</Notice>
        <DialogFooter><Button variant="outline" onClick={closeEditor} disabled={busy}>Cancel</Button><Button onClick={() => void saveLyrics()} loading={busy} disabled={ !lyrics.trim()}>{busy ? "Saving" : "Save lyrics"}</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  </div>;
}
