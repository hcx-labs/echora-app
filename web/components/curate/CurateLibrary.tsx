"use client";
import { publishCurations } from "../shell/sidebarCurations";

import { setUrl, useSubPath } from "../shell/urlState";
import Artwork from "../ui/Artwork";
import { toast } from "sonner";
import {
  Ellipsis,
  ListMusic,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  WandSparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { coverArtUrl } from "../media/coverArt";
import { mediaUrl } from "../media/mediaOrigin";
import { useEffect, useState, type ReactNode } from "react";
import { usePlayer, type PlayerTrack } from "../player/PlayerProvider";
import TagInput, { type Tag } from "./TagInput";
import TrackReferencePicker, { type ReferenceTrack } from "./TrackReferencePicker";
import { useCurationJobs } from "../jobs/useCurationJobs";
import LanguagePicker from "./LanguagePicker";
import SoundProfile, { type SoundProfileValues } from "./SoundProfile";
import { parseRecordingHandoff, resolveRecordingReference } from "./recordingHandoff";

import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Badge } from "../ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../ui/tabs";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "../ui/select";
import { Switch } from "../ui/switch";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../ui/dialog";
import Range from "./Range";
import { EmptyState } from "../layout/empty-state";
import { Pane, PaneSection, PaneSections } from "../layout/pane";
import { LoadingState } from "../ui/spinner";
import { Notice } from "../ui/notice";
import TransitionLink from "../shell/TransitionLink";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "../ui/sheet";

type Reference = { kind: "track" | "artist" | "album"; name: string };
type Track = {
  id: string;
  title: string;
  artist?: string;
  album?: string;
  duration_seconds: number;
  source_id: string;
  cover_art?: string;
  score: number;
  percentile?: number;
  position?: number;
  evidence?: { selection_pool?: "familiar" | "discovery"; listen_count?: number };
};
type Familiarity = {
  active: boolean;
  percent: number;
  familiar_tracks: number;
  discovery_tracks: number;
  matched_listens: number;
};
type LanguageReport = {
  target: string;
  strictness: "only" | "primarily" | "sprinkle";
  matched_tracks: number;
  requested: number;
};
type Preview = {
  tracks: Track[];
  references: { positive: Reference[]; negative: Reference[] };
  corpus_size: number;
  weights?: { semantic: number; lyrics: number };
  signal_weights?: Record<string, number>;
  familiarity?: Familiarity;
  language?: LanguageReport | null;
};
type StoredSoundProfile = SoundProfileValues & { instrumental_only?: boolean };
const CURATION_LANGUAGES = [
  ["", "Any language"],
  ["en", "English"],
  ["hi", "Hindi / Hinglish"],
  ["ur", "Urdu"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
  ["zh", "Mandarin"],
  ["es", "Spanish"],
  ["fr", "French"],
  ["de", "German"],
  ["pt", "Portuguese"],
  ["it", "Italian"],
  ["ru", "Russian"],
  ["ar", "Arabic"],
  ["id", "Indonesian"],
  ["th", "Thai"],
] as const;
const LANGUAGE_STRICTNESS = [
  ["only", "Only"],
  ["primarily", "Primarily"],
  ["sprinkle", "Sprinkle"],
] as const;
type RevisionRecipe = {
  references?: { positive: Reference[]; negative: Reference[] };
  weights?: { semantic: number; lyrics: number };
  signal_weights?: Record<string, number>;
  familiarity?: Familiarity;
  target_language?: string;
  language_strictness?: "only" | "primarily" | "sprinkle";
  sound_profile?: StoredSoundProfile;
  journey_start?: ReferenceTrack;
  journey_stops?: ReferenceTrack[];
  journey_end?: ReferenceTrack;
  journey_lyrics_weight?: number;
};
type CurationType = "combined" | "language" | "examples" | "time_of_day" | "sonic_journey";
type CurationAspect = "language" | "examples" | "time_of_day" | "sonic";
type Curation = {
  id: string;
  name: string;
  curation_type: CurationType;
  positive_prompt: string;
  negative_prompt: string;
  sound_prompts?: string[];
  themes_prompts?: string[];
  sound_negative_prompts?: string[];
  themes_negative_prompts?: string[];
  sound_weight?: number;
  sound_profile?: StoredSoundProfile;
  positive_tracks: ReferenceTrack[];
  negative_tracks: ReferenceTrack[];
  familiarity_percent: number;
  period_start?: string;
  period_end?: string;
  time_of_day_enabled?: boolean;
  lookback_days: number;
  track_limit: number;
  refresh_mode: "stable" | "fresh";
  refresh_enabled: boolean;
  status: string;
  last_error?: string;
  last_refreshed_at?: string;
  next_refresh_at?: string;
  recipe?: RevisionRecipe;
  tracks: Track[];
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/analysis${path}`, options);
  const text = response.status === 204 ? "" : await response.text();
  let body: { detail?: string } | null = null;
  if (text) {
    try {
      body = JSON.parse(text) as { detail?: string };
    } catch {
      body = null;
    }
  }
  if (!response.ok)
    throw new Error(body?.detail || text || `The request failed (${response.status})`);
  return (body ?? null) as T;
}

export default function CurateLibrary() {
  const { playQueue, track: current, playing, toggle } = usePlayer();
  const [connectionId, setConnectionId] = useState("");
  const [historyConnected, setHistoryConnected] = useState(false);
  const [name, setName] = useState("");
  const [activeAspect, setActiveAspect] = useState<CurationAspect>("language");
  const [positive, setPositive] = useState("");
  const [negative, setNegative] = useState("");
  const [soundTags, setSoundTags] = useState<Tag[]>([]);
  const [themeTags, setThemeTags] = useState<Tag[]>([]);
  const [soundWeight, setSoundWeight] = useState(50);
  const [soundProfile, setSoundProfile] = useState<SoundProfileValues>({});
  const [instrumentalOnly, setInstrumentalOnly] = useState(false);
  const [positiveTracks, setPositiveTracks] = useState<ReferenceTrack[]>([]);
  const [negativeTracks, setNegativeTracks] = useState<ReferenceTrack[]>([]);
  const [familiarityPercent, setFamiliarityPercent] = useState(70);
  const [periodStart, setPeriodStart] = useState("18:00");
  const [periodEnd, setPeriodEnd] = useState("23:00");
  const [timeEnabled, setTimeEnabled] = useState(false);
  const [trackLimit, setTrackLimit] = useState(30);
  const [refreshMode, setRefreshMode] = useState<"stable" | "fresh">("stable");
  const [targetLanguage, setTargetLanguage] = useState("");
  const [languageStrictness, setLanguageStrictness] = useState<"only" | "primarily" | "sprinkle">(
    "primarily",
  );
  const [journeyStart, setJourneyStart] = useState<ReferenceTrack[]>([]);
  const [journeyStops, setJourneyStops] = useState<ReferenceTrack[]>([]);
  const [journeyEnd, setJourneyEnd] = useState<ReferenceTrack[]>([]);
  const [journeyLyricsWeight, setJourneyLyricsWeight] = useState(25);
  const [refreshEnabled, setRefreshEnabled] = useState(true);
  const [pane, setPane] = useState<"build" | "songs" | "saved">("build");
  const [shapeOpen, setShapeOpen] = useState(true);
  const [savedOpen, setSavedOpen] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [detailKind, setDetailKind] = useState<"preview" | "saved" | null>(null);
  const [curations, setCurations] = useState<Curation[]>([]);
  const [curationsLoaded, setCurationsLoaded] = useState(false);
  const [busyAction, setBusyAction] = useState<"preview" | "save" | "refresh" | "delete" | null>(
    null,
  );
  const [deleteTarget, setDeleteTarget] = useState<Curation | null>(null);
  const [selectedCurationId, setSelectedCurationId] = useState<string | null>(null);
  // The open curation lives in the path (/curate/<id>): links, reloads and Back/Forward open it.
  const urlCurationId = useSubPath("curate");
  const [syncedUrlId, setSyncedUrlId] = useState<string | null | undefined>(undefined);
  const curationJobs = useCurationJobs(connectionId, async (signal) => {
    const body = await api<{ curations: Curation[] }>("/library/curations", { signal });
    if (signal.aborted) return;
    setCurations(body.curations);
    publishCurations(body.curations);
    const selected = body.curations.find((item) => item.id === selectedCurationId);
    if (selected && detailKind === "saved")
      setPreview((value) => (value ? { ...value, tracks: selected.tracks } : value));
  });
  const busy = busyAction !== null;

  function loadCurations() {
    return api<{ curations: Curation[] }>("/library/curations")
      .then((body) => {
        setCurations(body.curations);
        publishCurations(body.curations);
      })
      .catch((reason) => toast.error("Could not load curations", { description: reason.message }))
      .finally(() => setCurationsLoaded(true));
  }
  useEffect(() => {
    api<{ navidrome_connection_id?: string }>("/auth/me")
      .then((user) => setConnectionId(user.navidrome_connection_id || ""))
      .catch((reason) =>
        toast.error("Could not load your music connection", {
          description: reason instanceof Error ? reason.message : "Please sign in again.",
        }),
      );
    api<{ lastfm: { connected: boolean } }>("/settings")
      .then((body) => setHistoryConnected(body.lastfm.connected))
      .catch(() => {});
    loadCurations();
  }, []);

  useEffect(() => {
    const handoff = parseRecordingHandoff(window.location.search);
    if (!handoff) return;
    const controller = new AbortController();
    resolveRecordingReference(handoff.trackId, controller.signal)
      .then((track) => {
        if (controller.signal.aborted) return;
        if (handoff.intent === "journey") {
          setJourneyStart([track]);
          setActiveAspect("sonic");
        } else {
          setPositiveTracks([track]);
          setActiveAspect("examples");
        }
        // Remove only our consumed parameters, preserving Next's history state.
        const url = new URL(window.location.href);
        url.searchParams.delete("recording_track");
        url.searchParams.delete("recording_intent");
        window.history.replaceState(
          window.history.state,
          "",
          `${url.pathname}${url.search}${url.hash}`,
        );
      })
      .catch((reason) => {
        if (!controller.signal.aborted) fail("Could not load the recording", reason);
      });
    return () => controller.abort();
  }, []);

  const split = (tags: Tag[]) => ({
    positive: tags.filter((tag) => !tag.negative).map((tag) => tag.label),
    negative: tags.filter((tag) => tag.negative).map((tag) => tag.label),
  });
  const sound = split(soundTags);
  const themes = split(themeTags);
  const recipe = {
    curation_type: activeAspect === "sonic" ? "sonic_journey" : "combined",
    positive_prompt: positive,
    negative_prompt: negative,
    sound_prompts: sound.positive,
    themes_prompts: themes.positive,
    sound_negative_prompts: sound.negative,
    themes_negative_prompts: themes.negative,
    sound_weight: soundWeight,
    sound_profile: { ...soundProfile, ...(instrumentalOnly ? { instrumental_only: true } : {}) },
    positive_track_ids: positiveTracks.map((track) => track.id),
    negative_track_ids: negativeTracks.map((track) => track.id),
    familiarity_percent: familiarityPercent,
    period_start: timeEnabled ? periodStart : null,
    period_end: timeEnabled ? periodEnd : null,
    time_of_day_enabled: timeEnabled,
    lookback_days: 7,
    track_limit: trackLimit,
    refresh_mode: refreshMode,
    target_language: targetLanguage,
    language_strictness: languageStrictness,
    journey_start_track_id: journeyStart[0]?.id || null,
    journey_stop_track_ids: journeyStops.map((track) => track.id),
    journey_end_track_id: journeyEnd[0]?.id || null,
    journey_lyrics_weight: journeyLyricsWeight,
  };
  const hasSoundProfile = Object.keys(soundProfile).length > 0;
  const hasPositiveEvidence = Boolean(
    targetLanguage ||
    sound.positive.length ||
    themes.positive.length ||
    positive.trim() ||
    positiveTracks.length ||
    hasSoundProfile ||
    instrumentalOnly ||
    (timeEnabled && historyConnected && periodStart && periodEnd) ||
    (journeyStart.length && journeyEnd.length),
  );
  function fail(title: string, reason: unknown) {
    toast.error(title, {
      description: reason instanceof Error && reason.message !== title ? reason.message : undefined,
    });
  }
  async function generate() {
    setBusyAction("preview");
    try {
      setPreview(
        await api<Preview>("/library/curations/preview", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(recipe),
        }),
      );
      setDetailKind("preview");
      setPane("songs");
    } catch (reason) {
      fail("Could not generate the playlist", reason);
    } finally {
      setBusyAction(null);
    }
  }
  async function save() {
    if (!name.trim()) {
      toast.warning("Name the curation first", { description: "Give it a name before saving it." });
      return;
    }
    setBusyAction("save");
    try {
      const result = await api<{ job_id?: string; id?: string }>(
        selectedCurationId ? `/library/curations/${selectedCurationId}` : "/library/curations",
        {
          method: selectedCurationId ? "PUT" : "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ...recipe, name, refresh_enabled: refreshEnabled }),
        },
      );
      curationJobs.track(result?.job_id);
      await loadCurations();
      // A new curation becomes the open one, with its own URL, so later saves update it.
      if (!selectedCurationId && result?.id) {
        setSelectedCurationId(result.id);
        setDetailKind("saved");
        setSyncedUrlId(result.id);
        setUrl(`/curate/${result.id}`, "replace");
      }
      toast.success(selectedCurationId ? "Curation updated" : "Curation saved", {
        description: `${name.trim()} syncs to Navidrome shortly.`,
      });
    } catch (reason) {
      fail("Could not save the curation", reason);
    } finally {
      setBusyAction(null);
    }
  }
  function playerTracks(tracks: Track[]): PlayerTrack[] {
    return tracks.map((track) => ({
      id: track.id,
      title: track.title,
      artist: track.artist,
      album: track.album,
      durationSeconds: track.duration_seconds,
      coverUrl:
        track.cover_art && connectionId ? coverArtUrl(connectionId, track.cover_art) : undefined,
      streamUrl: mediaUrl(`/navidrome/connections/${connectionId}/stream/${track.source_id}`),
      connectionId,
      sourceId: track.source_id,
    }));
  }
  async function refresh(curation: Curation) {
    setBusyAction("refresh");
    setCurations((items) =>
      items.map((item) =>
        item.id === curation.id ? { ...item, status: "refreshing", last_error: undefined } : item,
      ),
    );
    try {
      const result = await api<{ job_id: string }>(`/library/curations/${curation.id}/refresh`, {
        method: "POST",
      });
      curationJobs.track(result.job_id);
    } catch (reason) {
      fail("Refresh failed", reason);
    } finally {
      loadCurations();
      setBusyAction(null);
    }
  }
  async function toggleSchedule(curation: Curation) {
    try {
      await api(`/library/curations/${curation.id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ refresh_enabled: !curation.refresh_enabled }),
      });
      loadCurations();
    } catch (reason) {
      fail("Could not update the schedule", reason);
    }
  }
  async function removeCuration(deleteNavidrome: boolean) {
    if (!deleteTarget) return;
    setBusyAction("delete");
    try {
      await api(`/library/curations/${deleteTarget.id}?delete_navidrome=${deleteNavidrome}`, {
        method: "DELETE",
      });
      if (deleteTarget.id === selectedCurationId) {
        closeDetails();
        setSyncedUrlId(null);
        setUrl("/curate", "replace");
      }
      setDeleteTarget(null);
      await loadCurations();
      toast.success("Curation deleted", { description: deleteTarget.name });
    } catch (reason) {
      fail("Could not delete the curation", reason);
    } finally {
      setBusyAction(null);
    }
  }
  function openCuration(curation: Curation) {
    setSelectedCurationId(curation.id);
    setName(curation.name);
    setActiveAspect(
      curation.curation_type === "sonic_journey"
        ? "sonic"
        : curation.time_of_day_enabled
          ? "time_of_day"
          : curation.positive_tracks?.length || curation.negative_tracks?.length
            ? "examples"
            : "language",
    );
    setPositive(curation.positive_prompt);
    setNegative(curation.negative_prompt);
    setSoundTags(
      (curation.sound_prompts || [])
        .map((label) => ({ label, negative: false }))
        .concat(
          (curation.sound_negative_prompts || []).map((label) => ({ label, negative: true })),
        ),
    );
    setThemeTags(
      (curation.themes_prompts || [])
        .map((label) => ({ label, negative: false }))
        .concat(
          (curation.themes_negative_prompts || []).map((label) => ({ label, negative: true })),
        ),
    );
    const savedSoundProfile = curation.recipe?.sound_profile || curation.sound_profile || {};
    const { instrumental_only, ...soundProfileValues } = savedSoundProfile;
    setSoundWeight(curation.sound_weight ?? 50);
    setSoundProfile(soundProfileValues);
    setInstrumentalOnly(Boolean(instrumental_only));
    setPositiveTracks(curation.positive_tracks || []);
    setNegativeTracks(curation.negative_tracks || []);
    setFamiliarityPercent(curation.familiarity_percent ?? 70);
    setPeriodStart(curation.period_start || "18:00");
    setPeriodEnd(curation.period_end || "23:00");
    setTimeEnabled(curation.time_of_day_enabled ?? curation.curation_type === "time_of_day");
    setTrackLimit(curation.track_limit);
    setRefreshMode(curation.refresh_mode);
    setRefreshEnabled(curation.refresh_enabled);
    setTargetLanguage(curation.recipe?.target_language || "");
    setLanguageStrictness(curation.recipe?.language_strictness || "primarily");
    setJourneyStart(curation.recipe?.journey_start ? [curation.recipe.journey_start] : []);
    setJourneyStops(curation.recipe?.journey_stops || []);
    setJourneyEnd(curation.recipe?.journey_end ? [curation.recipe.journey_end] : []);
    setJourneyLyricsWeight(curation.recipe?.journey_lyrics_weight ?? 25);
    setPreview({
      tracks: curation.tracks,
      references: curation.recipe?.references || { positive: [], negative: [] },
      weights: curation.recipe?.weights,
      signal_weights: curation.recipe?.signal_weights,
      familiarity: curation.recipe?.familiarity,
      corpus_size: 0,
    });
    setDetailKind("saved");
    setPane("build");
    setSavedOpen(false);
    setShapeOpen(true);
  }
  function resetRecipeForm() {
    setName("");
    setActiveAspect("language");
    setPositive("");
    setNegative("");
    setSoundTags([]);
    setThemeTags([]);
    setSoundWeight(50);
    setSoundProfile({});
    setInstrumentalOnly(false);
    setPositiveTracks([]);
    setNegativeTracks([]);
    setFamiliarityPercent(70);
    setPeriodStart("18:00");
    setPeriodEnd("23:00");
    setTimeEnabled(false);
    setTrackLimit(30);
    setRefreshMode("stable");
    setRefreshEnabled(true);
    setTargetLanguage("");
    setLanguageStrictness("primarily");
    setJourneyStart([]);
    setJourneyStops([]);
    setJourneyEnd([]);
    setJourneyLyricsWeight(25);
  }
  function closeDetails() {
    setSelectedCurationId(null);
    resetRecipeForm();
    setPane("build");
    setSavedOpen(false);
    setShapeOpen(true);
    setPreview(null);
    setDetailKind(null);
  }

  // Follow the URL: open the curation it names, or return to a new one. Adjusting
  // state during render (rather than in an effect) avoids a frame of the old view.
  if (curationsLoaded && urlCurationId !== syncedUrlId) {
    setSyncedUrlId(urlCurationId);
    const target = urlCurationId ? curations.find((item) => item.id === urlCurationId) : undefined;
    if (target) {
      if (target.id !== selectedCurationId) openCuration(target);
    } else if (selectedCurationId) closeDetails();
  }
  const missingCuration =
    curationsLoaded &&
    Boolean(urlCurationId) &&
    !curations.some((item) => item.id === urlCurationId);

  const shown = preview?.tracks || [];
  const recipeIncomplete =
    activeAspect === "sonic" ? !journeyStart.length || !journeyEnd.length : !hasPositiveEvidence;
  const shapeActive = Object.keys(soundProfile).length > 0 || instrumentalOnly;
  const selectedCuration = curations.find((item) => item.id === selectedCurationId);

  const savedList = !curationsLoaded ? (
    <LoadingState label="Loading curations…" />
  ) : curations.length ? (
    // minmax(0, 1fr): a long name truncates instead of widening the list past the pane.
    <ul className="grid grid-cols-[minmax(0,1fr)] gap-2">
      {curations.map((curation) => {
        const selected = selectedCurationId === curation.id;
        const refreshing =
          curation.status === "refreshing" ||
          curationJobs.jobs.some((job) => job.curation_id === curation.id);
        return (
          <li
            key={curation.id}
            className={cn(
              "relative min-w-0 border transition-colors",
              selected
                ? "border-primary/60 bg-primary/[.06]"
                : "border-border bg-surface/50 hover:border-border-strong hover:bg-surface",
            )}
          >
            <button
              type="button"
              disabled={busy}
              aria-current={selected ? "true" : undefined}
              onClick={() => setUrl(`/curate/${curation.id}`)}
              className="block w-full p-3 pr-20 text-left"
            >
              <strong
                className={cn("block truncate text-sm font-semibold", selected && "text-primary")}
              >
                {curation.name}
              </strong>
              <span className="mt-1 flex flex-wrap items-center gap-x-1.5 text-xs text-muted-foreground">
                <span>{curation.tracks.length} tracks</span>
                <span aria-hidden>·</span>
                <span
                  className={cn(
                    curation.status === "failed" && "text-destructive",
                    refreshing && "text-primary",
                  )}
                >
                  {refreshing ? "refreshing" : curation.status}
                </span>
                {curation.refresh_enabled && (
                  <>
                    <span aria-hidden>·</span>
                    <span>daily</span>
                  </>
                )}
              </span>
            </button>
            <div className="absolute top-2 right-2 flex">
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Refresh ${curation.name}`}
                disabled={busy || refreshing}
                onClick={() => refresh(curation)}
              >
                <RefreshCw className={refreshing ? "animate-spin" : undefined} />
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`More actions for ${curation.name}`}
                    disabled={busy}
                  >
                    <Ellipsis />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-52">
                  <DropdownMenuCheckboxItem
                    checked={curation.refresh_enabled}
                    onCheckedChange={() => toggleSchedule(curation)}
                  >
                    Refresh daily
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    variant="destructive"
                    onSelect={() => setDeleteTarget(curation)}
                  >
                    <Trash2 />
                    Delete…
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            {curation.last_error && (
              <p className="border-t border-destructive/30 px-3 py-2 text-xs text-destructive">
                {curation.last_error}
              </p>
            )}
            {curation.next_refresh_at && curation.refresh_enabled && !curation.last_error && (
              <p className="-mt-1 px-3 pb-3 text-xs text-subtle-foreground">
                Next{" "}
                {new Date(curation.next_refresh_at).toLocaleString(undefined, {
                  dateStyle: "medium",
                  timeStyle: "short",
                })}
              </p>
            )}
          </li>
        );
      })}
    </ul>
  ) : (
    <EmptyState
      icon={<ListMusic />}
      title="Nothing saved yet"
      description="Save a curation to sync it to Navidrome and keep it fresh."
    />
  );

  const builder = (
    <Column
      tone="build"
      title={selectedCuration ? selectedCuration.name : "New curation"}
      subtitle={selectedCuration ? "Editing saved curation" : "Shape a playlist from your library"}
      actions={
        <>
          {selectedCuration && (
            <Button variant="ghost" size="sm" disabled={busy} onClick={() => setUrl("/curate")}>
              <Plus />
              New
            </Button>
          )}
          <Sheet open={savedOpen} onOpenChange={setSavedOpen}>
            <SheetTrigger asChild>
              <Button variant="outline" size="sm" className="hidden md:inline-flex 2xl:hidden">
                <ListMusic />
                Saved
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[320px] gap-0 p-0">
              <SheetHeader className="sr-only">
                <SheetTitle>Saved curations</SheetTitle>
              </SheetHeader>
              <Column tone="saved" title="Saved" subtitle={`${curations.length} curations`}>
                {savedList}
              </Column>
            </SheetContent>
          </Sheet>
        </>
      }
      footer={
        <>
          <Button
            variant="outline"
            className="flex-1"
            onClick={generate}
            loading={busyAction === "preview"}
            disabled={busy || recipeIncomplete}
          >
            <WandSparkles />
            {busyAction === "preview" ? "Curating…" : "Preview"}
          </Button>
          <Button
            className="flex-1"
            onClick={save}
            loading={busyAction === "save"}
            disabled={busy || recipeIncomplete}
          >
            <Save />
            {busyAction === "save"
              ? "Saving…"
              : selectedCurationId
                ? "Save changes"
                : "Save + sync"}
          </Button>
        </>
      }
    >
      {missingCuration && (
        <Notice
          tone="warning"
          title="Curation not found"
          className="mb-4"
          action={
            <Button variant="outline" size="sm" onClick={() => setUrl("/curate", "replace")}>
              Start new
            </Button>
          }
        >
          It may have been deleted. You can build a new curation below.
        </Notice>
      )}
      <PaneSections key={selectedCurationId ?? "new"} className="motion-fade">
        <PaneSection title="Name">
          <Input
            id="playlist-name"
            aria-label="Curation name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Late-night drive"
          />
        </PaneSection>
        <PaneSection title="Criteria" hint="Combine any of these. Empty criteria are ignored.">
          <Tabs
            value={activeAspect}
            onValueChange={(value) => setActiveAspect(value as CurationAspect)}
            className="gap-5"
          >
            <TabsList variant="line" aria-label="Recipe criteria">
              <TabsTrigger value="language">Themes</TabsTrigger>
              <TabsTrigger value="examples">Examples</TabsTrigger>
              <TabsTrigger value="time_of_day">Period</TabsTrigger>
              <TabsTrigger value="sonic">Journey</TabsTrigger>
            </TabsList>
            <TabsContent value="language" className="grid gap-5">
              <div className="grid gap-2">
                <Label htmlFor="target-language">Language</Label>
                <LanguagePicker
                  id="target-language"
                  value={targetLanguage}
                  onChange={setTargetLanguage}
                  options={CURATION_LANGUAGES.map(([code, label]) => [code, label])}
                  ariaLabel="Target language"
                />
              </div>
              {targetLanguage && (
                <div className="grid gap-2">
                  <Label htmlFor="language-strictness">Strictness</Label>
                  <Select
                    value={languageStrictness}
                    onValueChange={(value) =>
                      setLanguageStrictness(value as typeof languageStrictness)
                    }
                  >
                    <SelectTrigger id="language-strictness" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {LANGUAGE_STRICTNESS.map(([value, label]) => (
                        <SelectItem key={value} value={value}>
                          {label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {languageStrictness === "only"
                      ? "Only confident language matches; the playlist may be shorter."
                      : languageStrictness === "primarily"
                        ? "Language matches first, then the closest musical matches."
                        : "All tracks compete, with a bonus for the language."}
                  </p>
                </div>
              )}
              <TagInput
                label="Sound"
                hint="instruments, production"
                placeholder="guitars, or -synths"
                tags={soundTags}
                onChange={setSoundTags}
              />
              <TagInput
                label="Themes"
                hint="lyrical meaning"
                placeholder="heartbreak, or -party"
                tags={themeTags}
                onChange={setThemeTags}
              />
              <p className="-mt-2 text-xs text-subtle-foreground">
                Press Enter to add. Start with − to exclude:{" "}
                <span className="text-success">+ include</span> ·{" "}
                <span className="text-destructive">− exclude</span>
              </p>
              {sound.positive.length > 0 && themes.positive.length > 0 && (
                <Range
                  label="Sound / themes blend"
                  value={soundWeight}
                  onChange={setSoundWeight}
                  detail={`${soundWeight}% sound · ${100 - soundWeight}% themes`}
                />
              )}
            </TabsContent>
            <TabsContent value="examples" className="grid gap-5">
              <TrackReferencePicker
                label="Songs like"
                value={positiveTracks}
                onChange={setPositiveTracks}
              />
              <TrackReferencePicker
                label="Songs not like"
                value={negativeTracks}
                onChange={setNegativeTracks}
              />
            </TabsContent>
            <TabsContent value="time_of_day" className="grid gap-4">
              <div className="flex items-center gap-3">
                <Switch
                  id="time-enabled"
                  checked={timeEnabled}
                  disabled={!historyConnected && !timeEnabled}
                  onCheckedChange={setTimeEnabled}
                />
                <Label htmlFor="time-enabled">Use a recurring listening period</Label>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="grid gap-2">
                  <Label htmlFor="period-start">From</Label>
                  <Input
                    id="period-start"
                    type="time"
                    disabled={!timeEnabled}
                    value={periodStart}
                    onChange={(event) => setPeriodStart(event.target.value)}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="period-end">Until</Label>
                  <Input
                    id="period-end"
                    type="time"
                    disabled={!timeEnabled}
                    value={periodEnd}
                    onChange={(event) => setPeriodEnd(event.target.value)}
                  />
                </div>
              </div>
              {historyConnected ? (
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Uses matched Last.fm listens from the past seven days. Overnight periods use your
                  Settings timezone.
                </p>
              ) : (
                <Notice tone="warning" title="Last.fm is not connected">
                  Connect listening history in{" "}
                  <TransitionLink href="/settings" className="text-primary hover:underline">
                    Settings
                  </TransitionLink>{" "}
                  to use listening periods.
                </Notice>
              )}
            </TabsContent>
            <TabsContent value="sonic" className="grid gap-5">
              <Notice tone="info" title="Journeys follow their waypoints">
                Only the start, stops, end and lyrics influence shape a journey. Other criteria and
                the sound shape are saved but not applied.
              </Notice>
              <TrackReferencePicker
                label="Start"
                value={journeyStart}
                onChange={setJourneyStart}
                single
              />
              <TrackReferencePicker
                label="Stops, in order"
                value={journeyStops}
                onChange={setJourneyStops}
              />
              <TrackReferencePicker
                label="End"
                value={journeyEnd}
                onChange={setJourneyEnd}
                single
              />
              <Range
                label="Lyrics influence"
                value={journeyLyricsWeight}
                max={60}
                onChange={setJourneyLyricsWeight}
                detail={`${100 - journeyLyricsWeight}% sound · ${journeyLyricsWeight}% lyrics`}
              />
            </TabsContent>
          </Tabs>
        </PaneSection>
        <PaneSection
          title={
            <span className="flex items-center gap-2">
              Sound shape
              {shapeActive && <span className="size-1.5 bg-primary" aria-label="active" />}
            </span>
          }
          hint={
            shapeActive
              ? "Targets applied relative to your library"
              : "Optional · pace, energy, vocals and more"
          }
          open={shapeOpen}
          onOpenChange={setShapeOpen}
        >
          {activeAspect === "sonic" && shapeActive && (
            <Notice tone="warning" className="mb-4">
              Sound shape is ignored while building a sonic journey.
            </Notice>
          )}
          <SoundProfile
            value={soundProfile}
            onChange={setSoundProfile}
            instrumentalOnly={instrumentalOnly}
            onInstrumentalOnlyChange={setInstrumentalOnly}
          />
        </PaneSection>
        <PaneSection title="Playlist" hint="Length, listening mix and refresh.">
          <div className="grid gap-5">
            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-2">
                <Label htmlFor="track-limit">Tracks</Label>
                <Input
                  id="track-limit"
                  type="number"
                  min={5}
                  max={200}
                  value={trackLimit}
                  onChange={(event) => setTrackLimit(Number(event.target.value))}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="refresh-style">Refresh style</Label>
                <Select
                  value={refreshMode}
                  onValueChange={(value) => setRefreshMode(value as typeof refreshMode)}
                >
                  <SelectTrigger id="refresh-style" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="stable">Stable</SelectItem>
                    <SelectItem value="fresh">Fresh</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid gap-1.5">
              <Range
                label="Listening mix"
                value={familiarityPercent}
                onChange={setFamiliarityPercent}
                detail={`${familiarityPercent}% familiar · ${100 - familiarityPercent}% new`}
              />
              {historyConnected ? (
                <p className="text-xs text-subtle-foreground">
                  Uses seven days of matched Last.fm listens.
                </p>
              ) : (
                <Notice tone="warning">
                  The mix is saved but has no effect until Last.fm is connected in{" "}
                  <TransitionLink href="/settings" className="text-primary hover:underline">
                    Settings
                  </TransitionLink>
                  .
                </Notice>
              )}
            </div>
            <div className="flex items-center gap-3">
              <Switch
                id="refresh-enabled"
                checked={refreshEnabled}
                onCheckedChange={setRefreshEnabled}
              />
              <Label htmlFor="refresh-enabled">Refresh every 24 hours</Label>
            </div>
          </div>
        </PaneSection>
      </PaneSections>
    </Column>
  );

  const songs = (
    <Column
      tone="songs"
      title={detailKind === "saved" ? "Songs" : preview ? "Preview" : "Songs"}
      subtitle={
        preview
          ? `${shown.length} tracks${detailKind === "preview" ? " · not saved" : ""}`
          : "Preview a recipe to see its songs"
      }
      actions={
        shown.length > 0 && (
          <Button size="sm" onClick={() => playQueue(playerTracks(shown), 0)}>
            <Play fill="currentColor" />
            Play all
          </Button>
        )
      }
      footer={
        preview && (preview.signal_weights || preview.familiarity?.active || preview.language) ? (
          <div className="space-y-0.5 text-xs text-muted-foreground">
            {preview.signal_weights && (
              <p>
                {Object.entries(preview.signal_weights)
                  .map(
                    ([signal, weight]) =>
                      `${Math.round(weight * 100)}% ${signal.replaceAll("_", " ")}`,
                  )
                  .join(" · ")}
              </p>
            )}
            {preview.familiarity?.active && (
              <p>
                {preview.familiarity.familiar_tracks} familiar ·{" "}
                {preview.familiarity.discovery_tracks} new
              </p>
            )}
            {preview.language && (
              <p>
                {preview.language.matched_tracks} {preview.language.target.toUpperCase()} matches ·{" "}
                {preview.language.strictness}
              </p>
            )}
          </div>
        ) : undefined
      }
    >
      {busyAction === "preview" ? (
        <LoadingState label="Curating your playlist…" className="py-24" />
      ) : !preview ? (
        <EmptyState
          icon={<WandSparkles />}
          title="No preview yet"
          description="Set some criteria in Build, then choose Preview. Or open a saved curation."
        />
      ) : (
        <>
          {preview.references.positive.length + preview.references.negative.length > 0 && (
            <div className="mb-4 flex flex-wrap gap-1.5">
              {(["positive", "negative"] as const).flatMap((kind) =>
                preview.references[kind].map((reference) => (
                  <Badge
                    variant="outline"
                    key={`${kind}-${reference.kind}-${reference.name}`}
                    className={kind === "negative" ? "text-destructive" : undefined}
                  >
                    {kind === "negative" ? "−" : "+"} {reference.kind}: {reference.name}
                  </Badge>
                )),
              )}
            </div>
          )}
          {shown.length ? (
            <ol
              key={`${detailKind}-${selectedCurationId ?? "preview"}-${shown.length}-${shown[0]?.id}`}
              className="motion-stagger"
            >
              {shown.map((track, index) => {
                const isCurrent = current?.id === track.id;
                return (
                  <li
                    key={track.id}
                    className={cn(
                      "group grid grid-cols-[32px_44px_minmax(0,1fr)_auto] items-center gap-3 border-b border-border px-2 py-2 last:border-0 hover:bg-surface",
                      isCurrent && "bg-raised",
                    )}
                  >
                    <Button
                      size="icon-sm"
                      variant="ghost"
                      aria-label={`${isCurrent && playing ? "Pause" : "Play"} ${track.title}`}
                      onClick={() => (isCurrent ? toggle() : playQueue(playerTracks(shown), index))}
                    >
                      {isCurrent && playing ? (
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
                    <Artwork
                      trackId={track.id}
                      sizes="44px"
                      src={
                        track.cover_art && connectionId
                          ? coverArtUrl(connectionId, track.cover_art, 88)
                          : undefined
                      }
                      className="size-11"
                    />
                    <div className="min-w-0">
                      <div
                        className={cn(
                          "truncate text-[13px] font-semibold",
                          isCurrent && "text-primary",
                        )}
                      >
                        {track.title}
                      </div>
                      <div className="truncate text-xs text-muted-foreground">
                        {track.artist || "Unknown artist"}
                        {track.album ? ` · ${track.album}` : ""}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {track.evidence?.selection_pool === "familiar" && (
                        <Badge
                          variant="secondary"
                          title={`${track.evidence.listen_count || 0} matched listens`}
                        >
                          Last.fm
                        </Badge>
                      )}
                      <span className="w-16 text-right text-xs tabular-nums text-muted-foreground">
                        {track.percentile == null
                          ? track.score.toFixed(2)
                          : `Top ${Math.max(1, Math.round((1 - track.percentile) * 100))}%`}
                      </span>
                    </div>
                  </li>
                );
              })}
            </ol>
          ) : (
            <EmptyState
              icon={<ListMusic />}
              title="No tracks yet"
              description="Preview the recipe, or wait for the saved curation to finish refreshing."
            />
          )}
        </>
      )}
    </Column>
  );

  const saved = (
    <Column
      tone="saved"
      title="Saved"
      subtitle={`${curations.length} ${curations.length === 1 ? "curation" : "curations"}`}
    >
      {savedList}
    </Column>
  );

  return (
    <>
      <div className="flex h-full min-h-0 flex-col">
        <div className="shrink-0 border-b border-border bg-rail p-2 md:hidden">
          <Tabs value={pane} onValueChange={(value) => setPane(value as typeof pane)}>
            <TabsList aria-label="Curation panes" className="w-full">
              <TabsTrigger value="build" className="flex-1">
                Build
              </TabsTrigger>
              <TabsTrigger value="songs" className="flex-1">
                Songs{preview ? ` · ${shown.length}` : ""}
              </TabsTrigger>
              <TabsTrigger value="saved" className="flex-1">
                Saved
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,1fr)] md:grid-cols-[minmax(400px,46%)_minmax(0,1fr)] xl:grid-cols-[minmax(460px,560px)_minmax(0,1fr)] 2xl:grid-cols-[580px_minmax(0,1fr)_320px]">
          <div
            className={cn(
              "min-h-0 min-w-0 border-border md:block md:border-r",
              pane === "build" ? "block" : "hidden",
            )}
          >
            {builder}
          </div>
          <div
            className={cn(
              "min-h-0 min-w-0 border-border md:block 2xl:border-r",
              pane === "songs" ? "block" : "hidden",
            )}
          >
            {songs}
          </div>
          <div
            className={cn(
              "min-h-0 min-w-0 md:hidden 2xl:block",
              pane === "saved" ? "block" : "hidden",
            )}
          >
            {saved}
          </div>
        </div>
      </div>
      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!open && !busy) setDeleteTarget(null);
        }}
      >
        <DialogContent
          onEscapeKeyDown={(event) => {
            if (busy) event.preventDefault();
          }}
          onInteractOutside={(event) => {
            if (busy) event.preventDefault();
          }}
        >
          <DialogHeader>
            <DialogTitle>Delete {deleteTarget?.name}?</DialogTitle>
            <DialogDescription>
              Choose whether to keep the synced playlist in Navidrome.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" disabled={busy} onClick={() => removeCuration(false)}>
              Delete from Echora only
            </Button>
            <Button
              variant="destructive"
              loading={busyAction === "delete"}
              disabled={busy}
              onClick={() => removeCuration(true)}
            >
              <Trash2 />
              Delete both
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

const toneClasses = { build: "bg-[#0e0e0e]", songs: "bg-workspace", saved: "bg-rail" } as const;

function Column({
  tone,
  ...props
}: {
  tone: keyof typeof toneClasses;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
}) {
  return <Pane {...props} className={toneClasses[tone]} />;
}
