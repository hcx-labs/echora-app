"use client";

import { Ellipsis, Info, ListPlus, Music2, Play, SquarePen, VolumeX, X } from "lucide-react";
import { useRef, useState } from "react";
import type { PlayerTrack } from "../player/PlayerProvider";
import { usePlayer } from "../player/PlayerProvider";
import { Button } from "../ui/button";
import { LanguagePicker } from "../ui/language-picker";
import { Textarea } from "../ui/textarea";
import { Notice } from "../ui/notice";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../ui/dialog";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "../ui/dropdown-menu";
import ProcessingDialog from "./ProcessingDialog";
import styles from "./TrackMenu.module.css";

type Track = {
  id: string;
  title: string;
  artist?: string;
  duration_seconds: number;
  source_id?: string;
  connectionId?: string;
  streamUrl?: string;
  coverUrl?: string;
  lyrics_status?: string;
  lyrics_text?: string;
  lyrics_language?: string;
};
type Lyrics = {
  text?: string | null;
  language?: string | null;
  availability_status?: string | null;
};

/** What changed about the song, for the list to show without reloading. */
export type TrackPatch = { lyrics_status?: string };

export default function TrackMenu({
  track,
  onSaved,
}: {
  track: Track;
  onSaved: (patch: TrackPatch) => void;
}) {
  const player = usePlayer();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  const [processingOpen, setProcessingOpen] = useState(false);
  const [lyrics, setLyrics] = useState("");
  const [language, setLanguage] = useState("");
  const [busy, setBusy] = useState(false);
  const closeEditor = () => {
    if (!busy) setEditorOpen(false);
  };
  const playable = Boolean(track.streamUrl && track.source_id);
  const playerTrack: PlayerTrack | null = playable
    ? {
        id: track.id,
        title: track.title,
        artist: track.artist,
        durationSeconds: track.duration_seconds,
        streamUrl: track.streamUrl!,
        coverUrl: track.coverUrl,
        connectionId: track.connectionId,
        sourceId: track.source_id,
      }
    : null;

  async function openEditor() {
    setOpen(false);
    setBusy(true);
    try {
      const response = await fetch(`/api/v1/library/tracks/${encodeURIComponent(track.id)}/lyrics`);
      const body = (await response.json()) as Lyrics;
      if (!response.ok) throw new Error("Could not load lyrics");
      setLyrics(body.text || "");
      setLanguage(body.language || "");
      setEditorOpen(true);
    } catch (reason) {
      toast.error("Could not load lyrics", {
        description:
          reason instanceof Error && reason.message !== "Could not load lyrics"
            ? reason.message
            : undefined,
      });
    } finally {
      setBusy(false);
    }
  }

  async function saveLyrics() {
    setBusy(true);
    try {
      const response = await fetch(
        `/api/v1/library/tracks/${encodeURIComponent(track.id)}/lyrics`,
        {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text: lyrics, language: language || null }),
        },
      );
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Could not save lyrics");
      setEditorOpen(false);
      window.dispatchEvent(new CustomEvent("echora:lyrics-update", { detail: track.id }));
      onSaved({ lyrics_status: "available" });
      toast.success("Lyrics saved", { description: track.title });
    } catch (reason) {
      toast.error("Could not save lyrics", {
        description:
          reason instanceof Error && reason.message !== "Could not save lyrics"
            ? reason.message
            : undefined,
      });
    } finally {
      setBusy(false);
    }
  }

  async function mark(status: "instrumental" | "missing") {
    setOpen(false);
    setBusy(true);
    try {
      const response = await fetch(
        `/api/v1/library/tracks/${encodeURIComponent(track.id)}/lyrics/status`,
        {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ status }),
        },
      );
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Could not update lyric status");
      window.dispatchEvent(new CustomEvent("echora:lyrics-update", { detail: track.id }));
      onSaved({ lyrics_status: status });
      toast.success(
        status === "instrumental" ? "Marked as instrumental" : "No longer marked as instrumental",
        { description: track.title },
      );
    } catch (reason) {
      toast.error("Could not update lyric status", {
        description:
          reason instanceof Error && reason.message !== "Could not update lyric status"
            ? reason.message
            : undefined,
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.root} onClick={(event) => event.stopPropagation()}>
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger asChild>
          <Button
            ref={triggerRef}
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label={`Actions for ${track.title}`}
            className="text-subtle-foreground hover:text-foreground"
          >
            <Ellipsis />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuItem
            disabled={!playerTrack}
            onSelect={() => {
              if (playerTrack) player.play(playerTrack);
            }}
          >
            <Play />
            Play now
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!playerTrack}
            onSelect={() => {
              if (playerTrack) player.playNext(playerTrack);
            }}
          >
            <ListPlus />
            Play next
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem disabled={busy} onSelect={() => void openEditor()}>
            <SquarePen />
            Edit lyrics
          </DropdownMenuItem>
          <DropdownMenuItem disabled={busy} onSelect={() => setProcessingOpen(true)}>
            <Info />
            Processing information
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={busy}
            onSelect={() =>
              void mark(track.lyrics_status === "instrumental" ? "missing" : "instrumental")
            }
          >
            {track.lyrics_status === "instrumental" ? <Music2 /> : <VolumeX />}
            {track.lyrics_status === "instrumental" ? "Mark non-instrumental" : "Mark instrumental"}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ProcessingDialog
        track={track}
        open={processingOpen}
        onOpenChange={setProcessingOpen}
        onCloseAutoFocus={() => triggerRef.current?.focus()}
      />
      <Dialog
        open={editorOpen}
        onOpenChange={(next) => {
          if (!busy) setEditorOpen(next);
        }}
      >
        <DialogContent
          className={`${styles.dialog} sm:max-w-[620px]`}
          showCloseButton={!busy}
          onCloseAutoFocus={() => triggerRef.current?.focus()}
        >
          <DialogHeader>
            <DialogTitle>Lyrics for {track.title}</DialogTitle>
            <DialogDescription>
              {track.artist || "Edit the lyrics and optional language."}
            </DialogDescription>
          </DialogHeader>
          <div className={styles.field}>
            <span>Language, optional</span>
            <div className={styles.languageRow}>
              <LanguagePicker
                aria-label="Lyrics language"
                placeholder="Identify from the lyrics"
                value={language}
                onChange={setLanguage}
              />
              {language && (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Clear the language"
                  onClick={() => setLanguage("")}
                >
                  <X />
                </Button>
              )}
            </div>
          </div>
          <label>
            Lyrics
            <Textarea
              className={styles.lyrics}
              value={lyrics}
              onChange={(event) => setLyrics(event.target.value)}
              placeholder="Paste or write the lyrics"
            />
          </label>
          <Notice tone="info">
            Saved lyrics skip AI transcription. Karaoke timing is regenerated during the next sync.
          </Notice>
          <DialogFooter>
            <Button variant="outline" onClick={closeEditor} disabled={busy}>
              Cancel
            </Button>
            <Button onClick={() => void saveLyrics()} loading={busy} disabled={!lyrics.trim()}>
              {busy ? "Saving" : "Save lyrics"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
