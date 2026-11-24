"use client";

import { setUrl } from "../shell/urlState";
import { libraryUrlSearch, parseLibraryUrl, type LibrarySort } from "./libraryUrl";
import {
  ArrowDown,
  ArrowUp,
  Search,
  SlidersHorizontal,
  Library,
  Play,
  Pause,
  Clock3,
  X,
} from "lucide-react";
import { coverArtUrl } from "../media/coverArt";
import { mediaUrl } from "../media/mediaOrigin";
import { useEffect, useRef, useState } from "react";
import { usePlayer } from "../player/PlayerProvider";
import { EmptyState } from "../layout/empty-state";
import { SidePanel } from "../layout/side-panel";
import { Pane, PaneSection, PaneSections, SectionAction } from "../layout/pane";
import { LoadingState, Spinner } from "../ui/spinner";
import FacetFilter from "./FacetFilter";
import Artwork from "../ui/Artwork";
import { Button } from "../ui/button";
import { Notice } from "../ui/notice";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "../ui/select";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "../ui/sheet";
import HumSearchButton from "./HumSearchButton";
import RecordingSearchButton from "./RecordingSearchButton";
import RecordingMatchNotice from "./RecordingMatchNotice";
import {
  canCurateRecording,
  curationHref,
  resultConnection,
  recordingQualityLabel,
  type RecordingQuality,
} from "./recordingSearch";
import TransitionLink from "../shell/TransitionLink";
import TrackMenu from "./TrackMenu";
import BrowseMetadataFilters, {
  emptyMetadataFilters,
  metadataTagGroups,
  type MetadataFilters,
  type MetadataTagKey,
} from "./BrowseMetadataFilters";
import { appendMetadataFilters, hasMetadataFilters } from "./metadataFilters";
import { loadLibraryFacets } from "./libraryFacets";

type Track = {
  connection_id?: string;
  id: string;
  title: string;
  artist?: string;
  album?: string;
  duration_seconds: number;
  source_id?: string;
  cover_art?: string;
  similarity?: number;
  matched_at_seconds?: number;
  matched_source?: string;
  lyrics_status?: string;
  recording_score?: number;
  recording_confirmed?: boolean;
  recording_quality?: RecordingQuality;
};
type Facet = { name: string; tracks: number };
const duration = (seconds: number) => {
  const rounded = Math.max(0, Math.round(seconds));
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`;
};

export default function BrowseLibrary() {
  const player = usePlayer();
  const recordingRef = useRef<{ cancel: () => void }>(null);
  const humRef = useRef<{ cancel: () => void }>(null);
  function invalidateSearches() {
    recordingRef.current?.cancel();
    humRef.current?.cancel();
  }
  const [tracks, setTracks] = useState<Track[]>([]);
  const [artists, setArtists] = useState<Facet[]>([]);
  const [facetsLoading, setFacetsLoading] = useState(true);
  const [albums, setAlbums] = useState<Facet[]>([]);
  const [total, setTotal] = useState(0);
  const [metadata, setMetadata] = useState<MetadataFilters>(emptyMetadataFilters);
  const [query, setQuery] = useState("");
  const [artist, setArtist] = useState("");
  const [album, setAlbum] = useState("");
  const [artistQuery, setArtistQuery] = useState("");
  const [albumQuery, setAlbumQuery] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [batch, setBatch] = useState(0);
  const [sortBy, setSortBy] = useState<LibrarySort>("name");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [selectedConnectionId, setConnectionId] = useState("");
  const [recordingWarning, setRecordingWarning] = useState("");
  const [searchMode, setSearchMode] = useState<"hum" | "recording" | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [urlRead, setUrlRead] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const pageSize = 50;

  useEffect(() => {
    // The address bar is the source for search, filters and sort, so links,
    // reloads and Back/Forward all restore the same view.
    const syncQuery = () => {
      const state = parseLibraryUrl(window.location.search);
      setQuery(state.q);
      setArtist(state.artist);
      setAlbum(state.album);
      setSortBy(state.sortBy);
      setSortDirection(state.sortDirection);
      setMetadata(state.metadata);
      setBatch(0);
      setUrlRead(true);
    };
    const search = (event: Event) => {
      setQuery((event as CustomEvent<string>).detail);
      setBatch(0);
      setSearchMode(null);
      setTracks([]);
    };
    syncQuery();
    window.addEventListener("popstate", syncQuery);
    window.addEventListener("echora:library-search", search);
    return () => {
      window.removeEventListener("popstate", syncQuery);
      window.removeEventListener("echora:library-search", search);
    };
  }, []);

  // Mirror the view back into the URL. Replace, not push: typing should not add history entries.
  useEffect(() => {
    if (!urlRead) return;
    const search = libraryUrlSearch({ q: query, artist, album, sortBy, sortDirection, metadata });
    if (search !== window.location.search)
      setUrl(`${window.location.pathname}${search}`, "replace");
  }, [urlRead, query, artist, album, sortBy, sortDirection, metadata]);

  useEffect(() => {
    fetch("/analysis/auth/me")
      .then((response) => (response.ok ? response.json() : null))
      .then((user) => setConnectionId(user?.navidrome_connection_id || ""))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (searchMode) return;
    const controller = new AbortController();
    const timer = window.setTimeout(
      () => {
        setLoading(true);
        setError("");
        const params = new URLSearchParams({
          limit: String(pageSize),
          offset: String(batch * pageSize),
          q: query,
          artist,
          album,
          sort_by: sortBy,
          sort_direction: sortDirection,
        });
        appendMetadataFilters(params, metadata);
        fetch(`/analysis/library/tracks?${params}`, { signal: controller.signal })
          .then(async (response) => {
            const body = await response.json();
            if (!response.ok) throw new Error(body.detail || "Could not load tracks");
            if (controller.signal.aborted) return;
            setTracks((current) =>
              batch === 0
                ? body.tracks
                : [
                    ...current,
                    ...body.tracks.filter(
                      (track: Track) => !current.some((item) => item.id === track.id),
                    ),
                  ],
            );
            setTotal(body.total);
          })
          .catch((reason) => {
            if (reason.name !== "AbortError")
              setError(reason instanceof Error ? reason.message : "Could not load tracks");
          })
          .finally(() => {
            if (!controller.signal.aborted) setLoading(false);
          });
      },
      batch === 0 ? 220 : 0,
    );
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [album, artist, batch, searchMode, query, refreshToken, sortBy, sortDirection, metadata]);

  const canLoadMore = !searchMode && !loading && tracks.length > 0 && tracks.length < total;
  function loadMore() {
    if (canLoadMore) setBatch((value) => value + 1);
  }
  // Page only in response to the reader scrolling near the end; never chain loads on our own.
  // A page that does not overflow cannot be scrolled; top it up only until it can.
  useEffect(() => {
    const root = listRef.current;
    if (root && root.clientHeight > 0 && canLoadMore && root.scrollHeight <= root.clientHeight + 1)
      setBatch((value) => value + 1);
  }, [canLoadMore, tracks.length]);
  function onListScroll() {
    const root = listRef.current;
    if (root && root.scrollHeight - root.scrollTop - root.clientHeight < 320) loadMore();
  }

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setFacetsLoading(true);
      const params = new URLSearchParams({
        artist_query: artistQuery,
        album_query: albumQuery,
        artist,
        limit: "20",
      });
      void loadLibraryFacets(
        params,
        controller.signal,
        (body) => {
          setArtists(body.artists);
          setAlbums(body.albums);
        },
        () => setFacetsLoading(false),
      );
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [albumQuery, artist, artistQuery]);

  function resetResults() {
    invalidateSearches();
    setLoading(true);
    setRefreshToken((value) => value + 1);
    setError("");
    setRecordingWarning("");
    setSearchMode(null);
    setTracks([]);
    setTotal(0);
    setBatch(0);
    listRef.current?.scrollTo({ top: 0 });
  }

  function showHumResults(results: Track[]) {
    setRecordingWarning("");
    setSearchMode("hum");
    setLoading(false);
    setTracks(results);
    setTotal(results.length);
    setBatch(0);
    setFiltersOpen(false);
    listRef.current?.scrollTo({ top: 0 });
  }

  function showRecordingResults(results: Track[], warning: string) {
    setRecordingWarning(warning);
    setSearchMode("recording");
    setLoading(false);
    setTracks(results);
    setTotal(results.length);
    setBatch(0);
    setFiltersOpen(false);
    listRef.current?.scrollTo({ top: 0 });
  }

  function playerTrack(track: Track) {
    const connectionId = resultConnection(track, selectedConnectionId);
    if (!connectionId || !track.source_id) return null;
    return {
      id: track.id,
      title: track.title,
      artist: track.artist,
      album: track.album,
      durationSeconds: track.duration_seconds,
      streamUrl: mediaUrl(
        `/navidrome/connections/${connectionId}/stream/${encodeURIComponent(track.source_id)}`,
      ),
      connectionId,
      sourceId: track.source_id,
      coverUrl: track.cover_art ? coverArtUrl(connectionId, track.cover_art) : undefined,
    };
  }
  function play(track: Track) {
    const next = playerTrack(track);
    if (next) player.play(next);
  }

  const hasFilters = Boolean(
    query || artist || album || searchMode || hasMetadataFilters(metadata),
  );
  const clearFilters = () => {
    setArtist("");
    setAlbum("");
    setArtistQuery("");
    setAlbumQuery("");
    setQuery("");
    setMetadata({ ...emptyMetadataFilters });
    resetResults();
  };
  const updateMetadata = (next: MetadataFilters) => {
    setMetadata(next);
    resetResults();
  };
  const filters = (
    <SidePanel
      title="Filters"
      actions={
        hasFilters ? (
          <Button variant="ghost" size="xs" onClick={clearFilters}>
            Reset
          </Button>
        ) : undefined
      }
    >
      <PaneSections>
        <PaneSection
          title="Artists"
          actions={
            artist && (
              <SectionAction
                onClick={() => {
                  setArtist("");
                  setAlbum("");
                  resetResults();
                }}
              >
                Clear
              </SectionAction>
            )
          }
        >
          <FacetFilter
            loading={facetsLoading}
            label="Artists"
            value={artist}
            query={artistQuery}
            items={artists}
            onQuery={(value) => {
              invalidateSearches();
              setArtistQuery(value);
            }}
            onSelect={(value) => {
              setArtist(value);
              setAlbum("");
              resetResults();
            }}
          />
        </PaneSection>
        <PaneSection
          title="Albums"
          actions={
            album && (
              <SectionAction
                onClick={() => {
                  setAlbum("");
                  resetResults();
                }}
              >
                Clear
              </SectionAction>
            )
          }
        >
          <FacetFilter
            loading={facetsLoading}
            label="Albums"
            value={album}
            query={albumQuery}
            items={albums}
            onQuery={(value) => {
              invalidateSearches();
              setAlbumQuery(value);
            }}
            onSelect={(value) => {
              setAlbum(value);
              resetResults();
            }}
          />
        </PaneSection>
        <BrowseMetadataFilters value={metadata} onChange={updateMetadata} />
      </PaneSections>
    </SidePanel>
  );
  type Chip = {
    key: string;
    label: string;
    group: "artist" | "album" | "year" | MetadataTagKey;
    value?: string;
  };
  const chips: Chip[] = [
    ...(artist ? [{ key: "artist", label: artist, group: "artist" as const }] : []),
    ...(album ? [{ key: "album", label: album, group: "album" as const }] : []),
    ...metadataTagGroups.flatMap((group) =>
      metadata[group.key].map((value) => ({
        key: `${group.key}:${value}`,
        label: `${group.label}: ${group.format(value)}`,
        group: group.key,
        value,
      })),
    ),
    ...(metadata.year_from || metadata.year_to
      ? [
          {
            key: "year",
            label: `Year: ${metadata.year_from || "…"}–${metadata.year_to || "…"}`,
            group: "year" as const,
          },
        ]
      : []),
  ];
  function removeChip(chip: Chip) {
    if (chip.group === "artist") {
      setArtist("");
      setAlbum("");
      resetResults();
    } else if (chip.group === "album") {
      setAlbum("");
      resetResults();
    } else if (chip.group === "year") updateMetadata({ ...metadata, year_from: "", year_to: "" });
    else
      updateMetadata({
        ...metadata,
        [chip.group]: metadata[chip.group].filter((item) => item !== chip.value),
      });
  }
  const activeTrack = (track: Track) => player.track?.id === track.id;
  return (
    <>
      <div className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)] md:grid-cols-[300px_minmax(0,1fr)]">
        <div className="hidden min-h-0 border-r border-border bg-rail md:block">{filters}</div>
        <Pane
          className="bg-workspace"
          label="Tracks"
          title={
            searchMode === "recording"
              ? "Recording results"
              : searchMode === "hum"
                ? "Melody matches"
                : "Your library"
          }
          subtitle={searchMode ? "Ranked by similarity" : undefined}
          actions={
            <>
              <span
                role="status"
                aria-label={
                  loading && !tracks.length
                    ? "Loading"
                    : `${total.toLocaleString()} ${searchMode ? (total === 1 ? "match" : "matches") : total === 1 ? "track" : "tracks"}`
                }
                className="mr-1 flex items-center text-lg leading-none font-semibold tracking-tight tabular-nums text-primary"
              >
                {loading && !tracks.length ? (
                  <Spinner className="size-5 text-primary" />
                ) : (
                  total.toLocaleString()
                )}
              </span>
              <Sheet open={filtersOpen} onOpenChange={setFiltersOpen}>
                <SheetTrigger asChild>
                  <Button variant="outline" className="md:hidden">
                    <SlidersHorizontal />
                    Filters
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-[300px] gap-0 p-0">
                  <SheetHeader className="sr-only">
                    <SheetTitle>Library filters</SheetTitle>
                  </SheetHeader>
                  {filters}
                </SheetContent>
              </Sheet>
            </>
          }
          toolbar={
            <>
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative min-w-56 flex-1">
                  <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-subtle-foreground" />
                  <Input
                    aria-label="Search tracks"
                    value={query}
                    onChange={(event) => {
                      setQuery(event.target.value);
                      resetResults();
                    }}
                    placeholder="Search songs, artists, albums"
                    className="pr-9 pl-9"
                  />
                  {loading && batch === 0 && !searchMode && (
                    <Spinner className="absolute top-1/2 right-3 -translate-y-1/2" />
                  )}
                </div>
                <HumSearchButton
                  ref={humRef}
                  onStart={invalidateSearches}
                  onResults={showHumResults}
                  onError={setError}
                />
                <RecordingSearchButton
                  ref={recordingRef}
                  onResults={showRecordingResults}
                  onError={setError}
                  onStart={invalidateSearches}
                />
                <div className="flex">
                  <Select
                    value={sortBy}
                    onValueChange={(value) => {
                      setSortBy(value as typeof sortBy);
                      resetResults();
                    }}
                  >
                    <SelectTrigger aria-label="Sort tracks" className="w-40">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="name">Name</SelectItem>
                      <SelectItem value="artist">Artist</SelectItem>
                      <SelectItem value="released">Released</SelectItem>
                      <SelectItem value="date_added">Recently added</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="icon"
                    className="-ml-px"
                    aria-label={`Sort ${sortDirection === "asc" ? "ascending" : "descending"}. Reverse order`}
                    onClick={() => {
                      setSortDirection((value) => (value === "asc" ? "desc" : "asc"));
                      resetResults();
                    }}
                  >
                    {sortDirection === "asc" ? <ArrowUp /> : <ArrowDown />}
                  </Button>
                </div>
              </div>
              {(chips.length > 0 || Boolean(searchMode)) && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <>
                    {searchMode && (
                      <Badge variant="outline">
                        {searchMode === "hum" ? "Hummed melody" : "Recorded audio"}
                      </Badge>
                    )}
                    {chips.map((chip) => (
                      <Badge key={chip.key} variant="secondary" className="h-7 gap-1 pr-1 text-xs">
                        {chip.label}
                        <button
                          type="button"
                          aria-label={`Remove ${chip.label}`}
                          onClick={() => removeChip(chip)}
                          className="grid size-5 place-items-center text-muted-foreground hover:bg-hover hover:text-foreground"
                        >
                          <X className="size-3" />
                        </button>
                      </Badge>
                    ))}
                    <Button variant="link" size="xs" onClick={clearFilters}>
                      Clear all
                    </Button>
                  </>
                </div>
              )}
            </>
          }
          scrollRef={listRef}
          onScroll={onListScroll}
          bodyClassName="pt-1"
        >
          {error && (
            <Notice tone="error" className="mt-4">
              {error}
            </Notice>
          )}
          {searchMode === "recording" && <RecordingMatchNotice message={recordingWarning} />}
          {loading && !tracks.length ? (
            <LoadingState
              label={hasFilters ? "Searching your library…" : "Loading your library…"}
              className="py-24"
            />
          ) : !tracks.length ? (
            <EmptyState
              icon={<Library />}
              title={hasFilters ? "No matching tracks" : "Bring your music home"}
              description={
                hasFilters
                  ? "Try another search or clear your filters."
                  : "Sync your Navidrome library to browse your collection and find your next listen."
              }
              actions={
                hasFilters ? (
                  <Button variant="outline" onClick={clearFilters}>
                    Clear filters
                  </Button>
                ) : (
                  <Button asChild>
                    <TransitionLink href="/sync">Sync your library</TransitionLink>
                  </Button>
                )
              }
            />
          ) : (
            <div className="motion-fade">
              <Table className="table-fixed">
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="w-14 pl-2 text-center">#</TableHead>
                    <TableHead>Title</TableHead>
                    <TableHead className="hidden w-[32%] lg:table-cell">Album</TableHead>
                    <TableHead className="w-16 text-right">
                      <Clock3 className="ml-auto size-3.5" />
                      <span className="sr-only">Duration</span>
                    </TableHead>
                    <TableHead className="w-12">
                      <span className="sr-only">Actions</span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tracks.map((track, index) => (
                    <TableRow
                      key={track.id}
                      data-state={activeTrack(track) ? "selected" : undefined}
                      className="group"
                    >
                      <TableCell className="pl-2 text-center">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          disabled={!playerTrack(track)}
                          aria-label={`${activeTrack(track) && player.playing ? "Pause" : "Play"} ${track.title}`}
                          onClick={() => (activeTrack(track) ? player.toggle() : play(track))}
                        >
                          {activeTrack(track) && player.playing ? (
                            <Pause className="size-4 text-primary" fill="currentColor" />
                          ) : (
                            <>
                              <span className="text-xs tabular-nums text-subtle-foreground group-hover:hidden">
                                {index + 1}
                              </span>
                              <Play
                                className="hidden size-4 text-foreground group-hover:block"
                                fill="currentColor"
                              />
                            </>
                          )}
                        </Button>
                      </TableCell>
                      <TableCell className="whitespace-normal">
                        <div className="flex min-w-0 items-center gap-3">
                          <Artwork
                            trackId={track.id}
                            src={playerTrack(track)?.coverUrl}
                            className="size-11 shrink-0"
                          />
                          <div className="min-w-0">
                            <strong
                              className={`block truncate text-[13px] font-semibold ${activeTrack(track) ? "text-primary" : ""}`}
                            >
                              {track.title}
                            </strong>
                            <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                              {track.artist ? (
                                <TransitionLink
                                  href={`/artists/${encodeURIComponent(track.artist)}`}
                                  className="hover:text-foreground hover:underline"
                                >
                                  {track.artist}
                                </TransitionLink>
                              ) : (
                                "Unknown artist"
                              )}
                              <span className="lg:hidden">
                                {track.album ? ` · ${track.album}` : ""}
                              </span>
                            </span>
                            {(track.similarity != null || track.matched_source === "recording") && (
                              <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                {track.similarity != null && (
                                  <Badge variant="outline">
                                    {Math.round(track.similarity * 100)}% match
                                  </Badge>
                                )}
                                {track.matched_source === "recording" && (
                                  <span>
                                    {recordingQualityLabel(track.recording_quality ?? "possible")} ·
                                    Score {track.recording_score?.toFixed(3)} · at{" "}
                                    {duration(track.matched_at_seconds || 0)}
                                  </span>
                                )}
                              </div>
                            )}
                            {canCurateRecording(track, index) && (
                              <div className="mt-1.5 flex flex-wrap gap-4 text-xs text-primary">
                                <TransitionLink
                                  className="hover:underline"
                                  href={curationHref(track.id, "similar")}
                                >
                                  More like this
                                </TransitionLink>
                                <TransitionLink
                                  className="hover:underline"
                                  href={curationHref(track.id, "journey")}
                                >
                                  Start journey
                                </TransitionLink>
                              </div>
                            )}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="hidden truncate text-muted-foreground lg:table-cell">
                        {track.album || "Unknown album"}
                      </TableCell>
                      <TableCell className="text-right text-xs tabular-nums text-muted-foreground">
                        {duration(track.duration_seconds)}
                      </TableCell>
                      <TableCell className="px-1">
                        <TrackMenu
                          track={{
                            ...track,
                            connectionId: resultConnection(track, selectedConnectionId),
                            streamUrl: playerTrack(track)?.streamUrl,
                            coverUrl: playerTrack(track)?.coverUrl,
                          }}
                          onSaved={() => {
                            if (searchMode) return;
                            setTracks([]);
                            setTotal(0);
                            setBatch(0);
                            setRefreshToken((value) => value + 1);
                          }}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
          {tracks.length > 0 && (
            <div className="flex justify-center pt-6 text-xs text-subtle-foreground">
              {loading ? (
                <span role="status" className="flex items-center gap-2">
                  <Spinner />
                  Loading more tracks…
                </span>
              ) : canLoadMore ? (
                <span>Scroll for more</span>
              ) : (
                !searchMode && <span>All {tracks.length.toLocaleString()} tracks shown</span>
              )}
            </div>
          )}
        </Pane>
      </div>
    </>
  );
}
