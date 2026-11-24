"use client";

import { useSearchParams } from "next/navigation";
import { setUrl } from "./shell/urlState";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  LibraryBig,
  ListMusic,
  Orbit,
  Pause,
  Play,
  RefreshCw,
  WandSparkles,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { EmptyState } from "./layout/empty-state";
import { Pane, PaneSection, PaneSections } from "./layout/pane";
import TransitionLink from "./shell/TransitionLink";
import { usePlayer, type PlayerTrack } from "./player/PlayerProvider";
import { coverArtUrl } from "./media/coverArt";
import { mediaUrl } from "./media/mediaOrigin";
import { getCachedUser } from "./session/sessionUser";
import Artwork from "./ui/Artwork";
import { Button } from "./ui/button";
import { Notice } from "./ui/notice";
import { LoadingState } from "./ui/spinner";
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs";
import { formatDuration } from "./ui/TrackRow";

type Track = {
  id: string;
  title: string;
  artist?: string;
  album?: string;
  duration_seconds: number;
  source_id?: string;
  cover_art?: string;
  play_count?: number;
};
type Curation = { id: string; name: string; tracks: Track[] };
type Period = "7day" | "1month" | "12month" | "overall";
// Last.fm ranks listening periods; Navidrome only has all-time play counts.
type TopTracks = { available: boolean; source?: "lastfm" | "navidrome" | null; tracks: Track[] };
const periods: [Period, string][] = [
  ["7day", "Week"],
  ["1month", "Month"],
  ["12month", "Year"],
  ["overall", "All time"],
];
const periodSlugs: [Period, string][] = [
  ["7day", "week"],
  ["1month", "month"],
  ["12month", "year"],
  ["overall", "all"],
];
const tools = [
  {
    title: "Explore your library",
    note: "Find a song, hum a melody, or match a recording.",
    href: "/library",
    Icon: LibraryBig,
  },
  {
    title: "Make a curation",
    note: "Turn a sound or a feeling into a living playlist.",
    href: "/curate",
    Icon: WandSparkles,
  },
  { title: "Enter the galaxy", note: "See how your songs connect.", href: "/galaxy", Icon: Orbit },
];

export default function HomeGrid() {
  const player = usePlayer();
  const [recent, setRecent] = useState<Track[]>([]);
  const [curations, setCurations] = useState<Curation[] | null>(null);
  const [connection, setConnection] = useState("");
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  // The Most played period is in the URL (/home?period=year) so it survives reloads.
  const periodSlug = useSearchParams().get("period");
  const period: Period = periodSlugs.find(([, slug]) => slug === periodSlug)?.[0] ?? "1month";
  const setPeriod = (next: Period) =>
    setUrl(
      next === "1month"
        ? "/home"
        : `/home?period=${periodSlugs.find(([value]) => value === next)?.[1]}`,
      "replace",
    );
  const [top, setTop] = useState<TopTracks | null>(null);
  const [topError, setTopError] = useState("");
  const [topLoading, setTopLoading] = useState(true);

  useEffect(() => {
    const abort = new AbortController();
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const user =
          getCachedUser() ||
          (await fetch("/analysis/auth/me", { signal: abort.signal }).then((response) =>
            response.ok ? response.json() : null,
          ));
        if (abort.signal.aborted) return;
        setConnection(user?.navidrome_connection_id || "");
        const response = await fetch(
          "/analysis/library/tracks?limit=10&offset=0&sort_by=date_added&sort_direction=desc",
          { signal: abort.signal },
        );
        if (!response.ok)
          throw new Error(
            "Could not load your library. Try again or check your connection in Settings.",
          );
        const body = await response.json();
        if (!abort.signal.aborted) {
          setRecent(body.tracks);
          setTotal(body.total);
        }
      } catch (reason) {
        if (!abort.signal.aborted)
          setError(reason instanceof Error ? reason.message : "Could not load your library.");
      } finally {
        if (!abort.signal.aborted) setLoading(false);
      }
    };
    load();
    fetch("/analysis/library/curations", { signal: abort.signal })
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (!abort.signal.aborted) setCurations(body?.curations || []);
      })
      .catch(() => {
        if (!abort.signal.aborted) setCurations([]);
      });
    return () => abort.abort();
  }, [retry]);

  useEffect(() => {
    const abort = new AbortController();
    const timer = window.setTimeout(() => {
      setTopLoading(true);
      setTopError("");
      fetch(`/analysis/library/top-tracks?limit=10&period=${period}`, { signal: abort.signal })
        .then(async (response) => {
          const body = await response.json().catch(() => null);
          if (!response.ok)
            throw new Error(body?.detail || "Most played tracks are unavailable right now.");
          return body as TopTracks;
        })
        .then((body) => {
          if (!abort.signal.aborted) setTop(body);
        })
        .catch((reason) => {
          if (!abort.signal.aborted && reason.name !== "AbortError")
            setTopError(
              reason instanceof Error
                ? reason.message
                : "Most played tracks are unavailable right now.",
            );
        })
        .finally(() => {
          if (!abort.signal.aborted) setTopLoading(false);
        });
    }, 0);
    return () => {
      window.clearTimeout(timer);
      abort.abort();
    };
  }, [period]);

  const cover = (track: Track, size = 320) =>
    track.cover_art && connection ? coverArtUrl(connection, track.cover_art, size) : undefined;
  const playable = (track: Track): PlayerTrack | null =>
    connection && track.source_id
      ? {
          id: track.id,
          title: track.title,
          artist: track.artist,
          album: track.album,
          durationSeconds: track.duration_seconds,
          streamUrl: mediaUrl(
            `/navidrome/connections/${connection}/stream/${encodeURIComponent(track.source_id)}`,
          ),
          connectionId: connection,
          sourceId: track.source_id,
          coverUrl: cover(track, 512),
        }
      : null;
  function playFrom(list: Track[], index: number) {
    const target = list[index];
    if (player.track?.id === target.id) {
      player.toggle();
      return;
    }
    const queue = list.map(playable).filter((item): item is PlayerTrack => Boolean(item));
    const start = queue.findIndex((item) => item.id === target.id);
    if (start >= 0) player.playQueue(queue, start);
  }
  const isPlaying = (track: Track) => player.track?.id === track.id && player.playing;
  const firstName = (getCachedUser()?.display_name || "").split(/\s+/)[0];

  const recentSection = (
    <PaneSection
      title="Recently added"
      hint="The newest tracks in your library."
      actions={
        <TransitionLink href="/library" className="text-xs text-primary hover:underline">
          Open library
        </TransitionLink>
      }
    >
      {loading ? (
        <div className="flex gap-4 overflow-hidden">
          {Array.from({ length: 6 }, (_, index) => (
            <div key={index} className="w-40 shrink-0">
              <div className="aspect-square animate-pulse bg-surface" />
              <div className="mt-2.5 h-3 w-3/4 bg-surface" />
              <div className="mt-1.5 h-3 w-1/2 bg-surface" />
            </div>
          ))}
        </div>
      ) : error ? (
        <Notice
          tone="error"
          action={
            <Button variant="outline" size="sm" onClick={() => setRetry((value) => value + 1)}>
              Try again
            </Button>
          }
        >
          {error}
        </Notice>
      ) : recent.length ? (
        <Shelf>
          {recent.map((track, index) => (
            <article key={track.id} className="group w-40 shrink-0 snap-start">
              <div className="relative">
                <Artwork
                  trackId={track.id}
                  src={cover(track)}
                  sizes="160px"
                  className="w-full border border-border"
                />
                <Button
                  size="icon"
                  className={cn(
                    "absolute right-2 bottom-2 size-10 shadow-lg shadow-black/50 transition-opacity",
                    isPlaying(track)
                      ? "opacity-100"
                      : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100 pointer-coarse:opacity-100",
                  )}
                  onClick={() => playFrom(recent, index)}
                  disabled={!playable(track)}
                  aria-label={`${isPlaying(track) ? "Pause" : "Play"} ${track.title}`}
                >
                  {isPlaying(track) ? <Pause fill="currentColor" /> : <Play fill="currentColor" />}
                </Button>
              </div>
              <strong
                className={cn(
                  "mt-2.5 block truncate text-[13px] font-semibold",
                  player.track?.id === track.id && "text-primary",
                )}
                title={track.title}
              >
                {track.title}
              </strong>
              <span className="block truncate text-xs text-muted-foreground">
                {track.artist || "Unknown artist"}
              </span>
            </article>
          ))}
        </Shelf>
      ) : (
        <EmptyState
          icon={<LibraryBig />}
          title="Your next discovery starts here"
          description="Sync your library to bring your music into Echora."
          actions={
            <Button asChild>
              <TransitionLink href="/sync">
                <RefreshCw />
                Sync library
              </TransitionLink>
            </Button>
          }
        />
      )}
    </PaneSection>
  );

  const topSection = (
    <PaneSection
      title="Most played"
      hint={
        top?.source === "navidrome"
          ? "All-time play counts from Navidrome. Connect Last.fm in Settings for weekly, monthly and yearly charts."
          : "Your top tracks from Last.fm, matched to your library."
      }
      actions={
        top?.source === "navidrome" ? undefined : (
          <Tabs value={period} onValueChange={(value) => setPeriod(value as Period)}>
            <TabsList aria-label="Listening period" className="h-8">
              {periods.map(([value, label]) => (
                <TabsTrigger key={value} value={value} className="px-2.5 text-xs">
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        )
      }
    >
      {topLoading && !top ? (
        <LoadingState label="Loading your most played tracks…" />
      ) : topError ? (
        <Notice tone="warning" title="Most played unavailable">
          {topError}
        </Notice>
      ) : top && !top.available ? (
        <Notice tone="info" title="No play counts yet">
          Connect Navidrome or Last.fm in{" "}
          <TransitionLink href="/settings" className="text-primary hover:underline">
            Settings
          </TransitionLink>{" "}
          to see your most played tracks.
        </Notice>
      ) : top && !top.tracks.length ? (
        <p className="py-8 text-center text-[13px] text-muted-foreground">
          {top.source === "navidrome"
            ? "Navidrome has no plays recorded for your library yet."
            : "No plays matched your library for this period."}
        </p>
      ) : (
        top && (
          <ol
            key={period}
            className={cn("motion-fade transition-opacity", topLoading && "opacity-50")}
          >
            {top.tracks.map((track, index) => {
              const current = player.track?.id === track.id;
              return (
                <li
                  key={track.id}
                  className={cn(
                    "group grid grid-cols-[32px_44px_minmax(0,1fr)_auto] items-center gap-3 border-b border-border px-2 py-2 last:border-0 hover:bg-surface md:grid-cols-[32px_44px_minmax(0,1.4fr)_minmax(0,1fr)_88px_48px]",
                    current && "bg-raised",
                  )}
                >
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    disabled={!playable(track)}
                    aria-label={`${isPlaying(track) ? "Pause" : "Play"} ${track.title}`}
                    onClick={() => playFrom(top.tracks, index)}
                  >
                    {isPlaying(track) ? (
                      <Pause className="text-primary" fill="currentColor" />
                    ) : (
                      <>
                        <span className="text-xs tabular-nums text-subtle-foreground group-hover:hidden">
                          {index + 1}
                        </span>
                        <Play className="hidden group-hover:block" fill="currentColor" />
                      </>
                    )}
                  </Button>
                  <Artwork trackId={track.id} src={cover(track, 96)} className="size-11" />
                  <div className="min-w-0">
                    <div
                      className={cn(
                        "truncate text-[13px] font-semibold",
                        current && "text-primary",
                      )}
                    >
                      {track.title}
                    </div>
                    <div className="truncate text-xs text-muted-foreground">
                      {track.artist || "Unknown artist"}
                    </div>
                  </div>
                  <div className="hidden truncate text-[13px] text-muted-foreground md:block">
                    {track.album || "Unknown album"}
                  </div>
                  <div className="text-right text-xs tabular-nums text-muted-foreground">
                    <span className="font-medium text-foreground">
                      {(track.play_count || 0).toLocaleString()}
                    </span>{" "}
                    plays
                  </div>
                  <div className="hidden text-right text-xs tabular-nums text-muted-foreground md:block">
                    {formatDuration(track.duration_seconds)}
                  </div>
                </li>
              );
            })}
          </ol>
        )
      )}
    </PaneSection>
  );

  const curationsSection = (
    <PaneSection
      title="Your curations"
      actions={
        curations?.length ? (
          <TransitionLink href="/curate" className="text-xs text-primary hover:underline">
            See all
          </TransitionLink>
        ) : undefined
      }
    >
      {curations === null ? (
        <LoadingState label="Loading curations…" className="min-h-0 py-8" />
      ) : curations.length ? (
        <div className="grid grid-cols-2 gap-4">
          {curations.slice(0, 4).map((curation) => (
            <TransitionLink
              href={`/curate/${curation.id}`}
              key={curation.id}
              className="group min-w-0"
            >
              <Collage
                covers={curation.tracks
                  .map((track) => cover(track, 120))
                  .filter((value): value is string => Boolean(value))
                  .slice(0, 9)}
              />
              <strong className="mt-2 block truncate text-[13px] font-semibold group-hover:text-primary">
                {curation.name}
              </strong>
              <small className="block text-xs text-muted-foreground">
                {curation.tracks.length} tracks
              </small>
            </TransitionLink>
          ))}
        </div>
      ) : (
        <TransitionLink
          href="/curate"
          className="flex items-center gap-3 border border-dashed border-border-strong p-4 hover:border-primary/60 hover:bg-surface"
        >
          <span className="grid size-10 shrink-0 place-items-center bg-raised text-primary">
            <ListMusic className="size-5" />
          </span>
          <span className="min-w-0 flex-1">
            <strong className="block text-[13px] font-semibold">Create your first curation</strong>
            <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">
              A mood, a melody, a late-night drive.
            </span>
          </span>
          <ArrowUpRight className="size-4 text-muted-foreground" />
        </TransitionLink>
      )}
    </PaneSection>
  );

  const jumpSection = (
    <PaneSection title="Jump to">
      <div className="divide-y divide-border border border-border">
        {tools.map(({ title, note, href, Icon }) => (
          <TransitionLink
            key={href}
            href={href}
            className="group flex items-center gap-3 p-3 transition-colors hover:bg-surface"
          >
            <span className="grid size-9 shrink-0 place-items-center bg-raised text-primary">
              <Icon className="size-4" />
            </span>
            <span className="min-w-0 flex-1">
              <strong className="block text-[13px] font-medium">{title}</strong>
              <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">
                {note}
              </span>
            </span>
            <ArrowUpRight className="size-4 text-subtle-foreground group-hover:text-foreground" />
          </TransitionLink>
        ))}
      </div>
    </PaneSection>
  );

  return (
    <>
      <div className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)] xl:grid-cols-[minmax(0,1fr)_360px]">
        <Pane
          className="bg-workspace"
          label="Home"
          title={firstName ? `Welcome back, ${firstName}` : "Welcome back"}
          subtitle="What's new in your library, and where to go next."
          actions={
            <>
              {total !== null && (
                <span
                  className="mr-1 text-lg leading-none font-semibold tracking-tight tabular-nums text-primary"
                  aria-label={`${total.toLocaleString()} tracks in your library`}
                  title="Tracks in your library"
                >
                  {total.toLocaleString()}
                </span>
              )}
              <Button asChild variant="outline" size="sm">
                <TransitionLink href="/sync">
                  <RefreshCw />
                  Sync
                </TransitionLink>
              </Button>
            </>
          }
        >
          <PaneSections className="pt-1">
            {recentSection}
            {topSection}
            <div className="contents xl:hidden">
              {curationsSection}
              {jumpSection}
            </div>
          </PaneSections>
        </Pane>
        <div className="hidden min-h-0 border-l border-border bg-rail xl:block">
          <Pane title="For you" subtitle="Your curations and shortcuts">
            <PaneSections>
              {curationsSection}
              {jumpSection}
            </PaneSections>
          </Pane>
        </div>
      </div>
    </>
  );
}

/** A horizontally scrolling row with paging buttons that appear only when there is more to see. */
function Shelf({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ start: true, end: true });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () =>
      setEdges({
        start: element.scrollLeft <= 1,
        end: element.scrollLeft + element.clientWidth >= element.scrollWidth - 1,
      });
    update();
    element.addEventListener("scroll", update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => {
      element.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, []);
  const page = (direction: 1 | -1) =>
    ref.current?.scrollBy({ left: direction * ref.current.clientWidth * 0.8, behavior: "smooth" });
  return (
    <div className="relative">
      <div ref={ref} className="motion-stagger flex snap-x gap-4 overflow-x-auto pb-3">
        {children}
      </div>
      {!edges.start && (
        <Button
          variant="outline"
          size="icon-sm"
          className="absolute top-[64px] -left-3 bg-workspace shadow-lg shadow-black/60"
          aria-label="Scroll back"
          onClick={() => page(-1)}
        >
          <ChevronLeft />
        </Button>
      )}
      {!edges.end && (
        <Button
          variant="outline"
          size="icon-sm"
          className="absolute top-[64px] -right-3 bg-workspace shadow-lg shadow-black/60"
          aria-label="Scroll forward"
          onClick={() => page(1)}
        >
          <ChevronRight />
        </Button>
      )}
    </div>
  );
}

/** Up to nine covers in a 3×3 grid; empty cells stay as quiet tiles. */
function Collage({ covers }: { covers: string[] }) {
  if (!covers.length) return <Artwork className="w-full border border-border" />;
  const cells =
    covers.length >= 9 ? covers : covers.length >= 4 ? covers.slice(0, 4) : covers.slice(0, 1);
  const columns =
    cells.length === 9 ? "grid-cols-3" : cells.length === 4 ? "grid-cols-2" : "grid-cols-1";
  return (
    <div
      className={cn(
        "grid aspect-square w-full gap-px overflow-hidden border border-border bg-border",
        columns,
      )}
    >
      {cells.map((src, index) => (
        <Artwork key={`${src}-${index}`} src={src} sizes="80px" className="size-full" />
      ))}
    </div>
  );
}
