"use client";

import { Maximize2, Minimize2, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { sizedPlayerCoverArtUrl } from "../media/coverArt";
import { FULLSCREEN_CLOSE_EVENT } from "../player/fullscreenEvents";
import { usePlayer } from "../player/PlayerProvider";
import VolumeControl from "../player/VolumeControl";
import WaveformSeek from "../player/WaveformSeek";
import Artwork from "../ui/Artwork";
import { Button } from "../ui/button";
import { Spinner } from "../ui/spinner";
import { formatDuration } from "../ui/TrackRow";

export default function MusicWidget() {
  const player = usePlayer();
  const hasNext = player.queueIndex >= 0 && player.queueIndex < player.queue.length - 1;
  // The fullscreen view owns its exit animation, so closing goes through it rather than straight to state.
  const toggleExpanded = () => {
    if (player.expanded) window.dispatchEvent(new Event(FULLSCREEN_CLOSE_EVENT));
    else player.setExpanded(true);
  };
  return (
    <section
      aria-label="Music player"
      className="grid h-full w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-4 px-4 md:grid-cols-[minmax(0,1fr)_minmax(0,min(820px,50%))_minmax(0,1fr)] md:gap-8 md:px-6"
    >
      <button
        type="button"
        disabled={!player.track}
        onClick={toggleExpanded}
        aria-expanded={player.expanded}
        className="flex min-w-0 items-center gap-3 text-left disabled:cursor-default"
        aria-label={
          !player.track
            ? "No track selected"
            : player.expanded
              ? "Close full screen player"
              : `Open player for ${player.track.title}`
        }
      >
        <Artwork
          src={
            player.track?.coverUrl ? sizedPlayerCoverArtUrl(player.track.coverUrl, 112) : undefined
          }
          className="size-12 shrink-0 border border-border max-md:size-11"
        />
        <span className="min-w-0">
          <strong className="block truncate text-[13px] font-semibold">
            {player.track?.title || "Nothing playing"}
          </strong>
          <span className="mt-0.5 flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
            {player.buffering ? (
              <span>Preparing audio…</span>
            ) : player.track ? (
              <span className="truncate">
                {player.track.artist || "Unknown artist"}
                {player.track.album ? ` · ${player.track.album}` : ""}
              </span>
            ) : (
              <span>Choose a track to begin</span>
            )}
          </span>
        </span>
      </button>
      <div className="flex min-w-0 items-center gap-4">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            className="max-sm:hidden"
            onClick={player.previous}
            disabled={!player.track}
            aria-label="Previous track"
          >
            <SkipBack />
          </Button>
          <Button
            size="icon"
            className="size-9"
            onClick={player.toggle}
            disabled={!player.track}
            aria-label={player.buffering ? "Loading audio" : player.playing ? "Pause" : "Play"}
            aria-busy={player.buffering || undefined}
          >
            {player.buffering ? (
              <Spinner className="text-current" />
            ) : player.playing ? (
              <Pause fill="currentColor" />
            ) : (
              <Play fill="currentColor" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={player.next}
            disabled={!hasNext}
            aria-label="Next track"
          >
            <SkipForward />
          </Button>
        </div>
        <div className="hidden min-w-0 flex-1 items-center gap-2.5 md:flex">
          <time className="w-9 text-right text-xs tabular-nums text-muted-foreground">
            {formatDuration(player.currentTime)}
          </time>
          <div className="min-w-0 flex-1">
            <WaveformSeek compact />
          </div>
          <time className="w-9 text-xs tabular-nums text-muted-foreground">
            {formatDuration(player.duration)}
          </time>
        </div>
      </div>
      <div className="hidden items-center justify-end gap-1 md:flex">
        <VolumeControl />
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleExpanded}
          disabled={!player.track}
          aria-pressed={player.expanded}
          aria-label={player.expanded ? "Close full screen player" : "Open full screen player"}
        >
          {player.expanded ? (
            <Minimize2 className="size-[18px]" />
          ) : (
            <Maximize2 className="size-[18px]" />
          )}
        </Button>
      </div>
    </section>
  );
}
