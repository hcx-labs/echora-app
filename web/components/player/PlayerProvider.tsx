"use client";

import { createContext, ReactNode, useContext, useEffect, useRef, useState } from "react";
import { sizedPlayerCoverArtUrl } from "../media/coverArt";
import { paletteFromPixels, panelColorFromPalette, type TrackPalette } from "./artworkPalette";
import { publishPlaybackTime, smoothPlaybackTime, type ClockAnchor } from "./playbackClock";
import FullscreenPlayer from "./FullscreenPlayer";
import { readPlaybackPreferences, streamUrlForQuality } from "./playbackPreferences";
import {
  descriptorRhythm,
  validateVisualEnrichment,
  validateVisualFeatures,
  visualFrameAt,
  neutralVisualFrame,
  publishVisualFrame,
  type VisualFeatureTimeline,
} from "./visualFeatures";
import { mediaUrl } from "../media/mediaOrigin";
import { StemMixer, stemsSupported } from "./karaokeStems";
import { SESSION_EXPIRED_EVENT } from "../session/sessionUser";

export type PlayerTrack = {
  id: string;
  title: string;
  artist?: string;
  album?: string;
  durationSeconds?: number;
  streamUrl: string;
  coverUrl?: string;
  connectionId?: string;
  sourceId?: string;
};

function scrobble(track: PlayerTrack | null, submission: boolean) {
  if (!track?.connectionId || !track.sourceId) return;
  fetch(mediaUrl("/media/scrobble"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      connection_id: track.connectionId,
      song_id: track.sourceId,
      submission,
    }),
  }).catch(() => {});
}
export type AudioQuality = {
  codec?: string;
  content_type?: string;
  bit_rate_kbps?: number;
  bit_depth?: number;
  sample_rate_hz?: number;
  channels?: number;
  lossless?: boolean;
  streamQuality: "original" | "320" | "120";
};
export type LyricTranslation = {
  source_language: string;
  target_language: string;
  source_lines: string[];
  lines: { id: number; text: string }[];
};
export type PlayerLyrics = {
  translations?: LyricTranslation[];
  trackId: string;
  available: boolean;
  karaoke?: boolean;
  /** "fa_kara" when Echora's aligner timed the words; "navidrome" when they came with the lyrics. */
  karaoke_model?: string | null;
  lines?: {
    start_ms: number | null;
    end_ms?: number;
    text: string;
    syllables?: { start_ms: number; end_ms: number; text: string; voice?: number }[];
  }[];
  text?: string;
  language?: string;
  provenance?: {
    ai_generated?: boolean;
    synced?: boolean;
    lines?: {
      start_ms: number | null;
      end_ms?: number;
      text: string;
      syllables?: { start_ms: number; end_ms: number; text: string; voice?: number }[];
    }[];
  };
};
export type MelodyPreview = {
  source: string;
  points: { time_seconds: number; pitch: number | null }[];
};

/** The Stage player's karaoke stems: the song's vocals and instrumental, remixed live. */
export type KaraokeStemsState = {
  /** The browser can play and remix the stems file. */
  supported: boolean;
  /** The current song has stems. */
  available: boolean;
  /** The listener wants stems (kept across songs and visits). */
  enabled: boolean;
  /** The player is streaming the stems now, not the original song. */
  playing: boolean;
  /** -1 vocals only, 0 the full song, 1 instrumental only. */
  balance: number;
  setEnabled: (value: boolean) => void;
  setBalance: (value: number) => void;
};

type PlayerState = {
  track: PlayerTrack | null;
  karaokeStems: KaraokeStemsState;
  audioQuality: AudioQuality | null;
  lyrics: PlayerLyrics | null;
  lyricsLoading: boolean;
  playing: boolean;
  buffering: boolean;
  currentTime: number;
  duration: number;
  buffered: number;
  muted: boolean;
  volume: number;
  expanded: boolean;
  waveform: number[] | null;
  melody: MelodyPreview | null;
  queue: PlayerTrack[];
  queueIndex: number;
  play: (track: PlayerTrack) => void;
  playQueue: (tracks: PlayerTrack[], startIndex?: number) => void;
  playNext: (track: PlayerTrack) => void;
  next: () => void;
  previous: () => void;
  clearQueue: () => void;
  toggle: () => void;
  seek: (seconds: number) => void;
  toggleMute: () => void;
  setVolume: (value: number) => void;
  setExpanded: (value: boolean) => void;
};

const PlayerContext = createContext<PlayerState | null>(null);
const volumeStorageKey = "echora:volume";
const stemsEnabledStorageKey = "echora:karaoke-stems";
const stemsBalanceStorageKey = "echora:karaoke-stems-balance";
// Waiting longer than this for the stems lookup plays the original song instead.
const STEMS_LOOKUP_TIMEOUT_MS = 3000;

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function writeStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage may be unavailable */
  }
}

const rgb = (color: [number, number, number]) => `rgb(${color.join(" ")})`;

function publishPalette(palette: TrackPalette | null) {
  const root = document.documentElement;
  if (palette) {
    root.style.setProperty("--surface-rgb", panelColorFromPalette(palette).join(" "));
    // The app accent follows the playing artwork. paletteFromPixels keeps it light
    // enough for dark panels and for the dark text on accent fills.
    root.style.setProperty("--accent", rgb(palette.accent));
    root.style.setProperty("--accent-rgb", palette.accent.join(" "));
  } else {
    // Nothing playing: fall back to the Echora blue defined in globals.css.
    root.style.removeProperty("--surface-rgb");
    root.style.removeProperty("--accent");
    root.style.removeProperty("--accent-rgb");
  }
  window.dispatchEvent(
    new CustomEvent("echora:track-palette", { detail: { active: Boolean(palette), palette } }),
  );
}

function hasMediaSession() {
  return typeof navigator !== "undefined" && "mediaSession" in navigator;
}

function publishMediaMetadata(track: PlayerTrack) {
  if (!hasMediaSession() || typeof MediaMetadata === "undefined") return;
  const artwork = track.coverUrl
    ? [{ src: new URL(sizedPlayerCoverArtUrl(track.coverUrl, 512), window.location.href).href }]
    : [];
  navigator.mediaSession.metadata = new MediaMetadata({
    title: track.title || "Unknown title",
    artist: track.artist || "Unknown artist",
    album: track.album || "",
    artwork,
  });
}

function publishMediaPosition(player: HTMLAudioElement, fallbackDuration = 0) {
  if (!hasMediaSession() || !navigator.mediaSession.setPositionState) return;
  const duration =
    Number.isFinite(player.duration) && player.duration > 0 ? player.duration : fallbackDuration;
  if (!Number.isFinite(duration) || duration <= 0) return;
  navigator.mediaSession.setPositionState({
    duration,
    playbackRate: player.playbackRate || 1,
    position: Math.min(Math.max(player.currentTime || 0, 0), duration),
  });
}

async function artworkPalette(url: string): Promise<TrackPalette> {
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.src = url;
  await image.decode();
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas is unavailable");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, size, size);
  const pixels = context.getImageData(0, 0, size, size).data;
  return paletteFromPixels(pixels);
}

export function PlayerProvider({ children }: { children: ReactNode }) {
  const audio = useRef<HTMLAudioElement | null>(null);
  const visualFeatures = useRef<VisualFeatureTimeline | null>(null);
  const analysisFrame = useRef(0);
  const visualRequest = useRef<AbortController | null>(null);
  const previousVisualTime = useRef<number | null>(null);
  const [loadVersion, setLoadVersion] = useState(0);
  function resetVisuals() {
    previousVisualTime.current = null;
    publishVisualFrame(neutralVisualFrame(audio.current?.currentTime || 0));
  }
  const queueRef = useRef<PlayerTrack[]>([]);
  const trackRef = useRef<PlayerTrack | null>(null);
  const paletteRef = useRef<TrackPalette | null>(null);
  const paletteTrackRef = useRef("");
  const playbackSessionRef = useRef(0);
  const lyricsCacheRef = useRef(new Map<string, PlayerLyrics>());
  const lyricsGenerationRef = useRef(new Map<string, number>());
  const queueIndexRef = useRef(-1);
  const activateQueueIndexRef = useRef<(index: number) => void>(() => {});
  const nextRef = useRef<() => void>(() => {});
  const previousRef = useRef<() => void>(() => {});
  const listenedRef = useRef(0);
  const announcedRef = useRef("");
  const [track, setTrack] = useState<PlayerTrack | null>(null);
  const [waveformData, setWaveformData] = useState<{
    trackId: string;
    peaks: number[] | null;
    melody: MelodyPreview | null;
  } | null>(null);
  const waveform = waveformData?.trackId === track?.id ? (waveformData?.peaks ?? null) : null;
  const melody = waveformData?.trackId === track?.id ? (waveformData?.melody ?? null) : null;
  useEffect(() => {
    if (!track?.id) return;
    const trackId = track.id;
    visualFeatures.current = null;
    const controller = new AbortController();
    visualRequest.current = controller;
    resetVisuals();
    fetch(`/api/v1/library/tracks/${encodeURIComponent(trackId)}/visual-features`, {
      signal: controller.signal,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (
          controller.signal.aborted ||
          visualRequest.current !== controller ||
          trackRef.current?.id !== trackId
        )
          return;
        if (body?.track_id !== trackId) return;
        const envelope = body?.visual_features;
        visualFeatures.current = validateVisualFeatures(
          envelope?.features ? (envelope.revision === "2" ? envelope.features : null) : envelope,
        );
        if (visualFeatures.current) {
          visualFeatures.current.rhythm = descriptorRhythm(
            body?.enrichment,
            visualFeatures.current.duration_seconds,
          );
          visualFeatures.current.enrichment = validateVisualEnrichment(
            body?.enrichment,
            visualFeatures.current.duration_seconds,
          );
        }
        resetVisuals();
      })
      .catch(() => {});
    fetch(`/api/v1/library/tracks/${encodeURIComponent(trackId)}/waveform`, {
      signal: controller.signal,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        const peaks = body?.waveform?.peaks;
        const validPeaks =
          Array.isArray(peaks) &&
          peaks.length > 0 &&
          peaks.length <= 1024 &&
          peaks.every(
            (value) =>
              typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1,
          );
        const melody = body?.melody;
        const validMelody =
          melody &&
          typeof melody.source === "string" &&
          Array.isArray(melody.points) &&
          melody.points.length <= 1024 &&
          melody.points.every(
            (point: { time_seconds?: unknown; pitch?: unknown }) =>
              typeof point.time_seconds === "number" &&
              Number.isFinite(point.time_seconds) &&
              point.time_seconds >= 0 &&
              (point.pitch === null ||
                (typeof point.pitch === "number" && Number.isFinite(point.pitch))),
          );
        if (
          !controller.signal.aborted &&
          visualRequest.current === controller &&
          trackRef.current?.id === trackId
        )
          setWaveformData({
            trackId,
            peaks: validPeaks ? peaks : null,
            melody: validMelody ? melody : null,
          });
      })
      .catch(() => {});
    return () => controller.abort();
  }, [track?.id, loadVersion]);
  const [audioQuality, setAudioQuality] = useState<AudioQuality | null>(null);
  const [lyrics, setLyrics] = useState<PlayerLyrics | null>(null);
  const [lyricsLoading, setLyricsLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [buffered, setBuffered] = useState(0);
  const [muted, setMuted] = useState(false);
  const [volume, setVolumeState] = useState(1);
  const [expanded, setExpanded] = useState(false);
  const stemMixer = useRef<StemMixer | null>(null);
  const stemsEnabledRef = useRef(false);
  const stemsRef = useRef<{ trackId: string; url: string } | null>(null);
  // Known after mount: the server render has no Audio to ask.
  const [stemsSupport, setStemsSupport] = useState(false);
  const [stems, setStems] = useState<{ trackId: string; url: string } | null>(null);
  const [stemsEnabled, setStemsEnabledState] = useState(false);
  const [stemsBalance, setStemsBalanceState] = useState(1);
  const [stemsPlaying, setStemsPlaying] = useState(false);
  const [queue, setQueue] = useState<PlayerTrack[]>([]);
  const [queueIndex, setQueueIndex] = useState(-1);

  useEffect(() => {
    const player = new Audio();
    audio.current = player;
    const mixer = new StemMixer(player);
    stemMixer.current = mixer;
    const storedBalance = Number(readStorage(stemsBalanceStorageKey));
    const balance =
      readStorage(stemsBalanceStorageKey) !== null && Number.isFinite(storedBalance)
        ? Math.min(1, Math.max(-1, storedBalance))
        : 1;
    mixer.setBalance(balance);
    stemsEnabledRef.current = readStorage(stemsEnabledStorageKey) === "true";
    const supported = stemsSupported();
    queueMicrotask(() => {
      setStemsSupport(supported);
      setStemsBalanceState(balance);
      setStemsEnabledState(stemsEnabledRef.current);
    });
    const storedVolume = Number(localStorage.getItem(volumeStorageKey));
    if (localStorage.getItem(volumeStorageKey) !== null && Number.isFinite(storedVolume)) {
      player.volume = Math.min(1, Math.max(0, storedVolume));
      queueMicrotask(() => setVolumeState(player.volume));
    }
    // Native media events continue in background tabs, unlike animation frames.
    const sessionPlayback = () =>
      window.dispatchEvent(
        new CustomEvent("echora:session-playback", {
          detail: {
            source: player.currentSrc,
            currentTime: player.currentTime,
            paused: player.paused || player.ended,
            seeking: player.seeking,
            readyState: player.readyState,
          },
        }),
      );
    const sessionEvents = [
      "timeupdate",
      "pause",
      "seeking",
      "emptied",
      "waiting",
      "ended",
    ] as const;
    sessionEvents.forEach((event) => player.addEventListener(event, sessionPlayback));
    const sessionExpired = () => {
      // Invalidate in-flight work before clearing caches, including same-track
      // requests that could otherwise publish after another account signs in.
      playbackSessionRef.current++;
      trackRef.current = null;
      queueRef.current = [];
      queueIndexRef.current = -1;
      lyricsCacheRef.current.clear();
      lyricsGenerationRef.current.clear();
      paletteRef.current = null;
      paletteTrackRef.current = "";
      visualRequest.current?.abort();
      visualRequest.current = null;
      visualFeatures.current = null;
      listenedRef.current = 0;
      announcedRef.current = "";
      player.pause();
      player.removeAttribute("src");
      player.load();
      cancelAnimationFrame(analysisFrame.current);
      analysisFrame.current = 0;
      resetVisuals();
      publishPlaybackTime(0);
      publishPalette(null);
      setTrack(null);
      setQueue([]);
      setQueueIndex(-1);
      setLyrics(null);
      setLyricsLoading(false);
      setAudioQuality(null);
      stemsRef.current = null;
      setStems(null);
      setStemsPlaying(false);
      setWaveformData(null);
      setCurrentTime(0);
      setDuration(0);
      setBuffered(0);
      setPlaying(false);
      setBuffering(false);
      setExpanded(false);
      window.dispatchEvent(new CustomEvent("echora:track-change", { detail: null }));
      window.dispatchEvent(new CustomEvent("echora:playback-state", { detail: false }));
      if (hasMediaSession()) {
        navigator.mediaSession.metadata = null;
        navigator.mediaSession.playbackState = "none";
        navigator.mediaSession.setPositionState?.();
      }
    };
    window.addEventListener(SESSION_EXPIRED_EVENT, sessionExpired);
    const time = () => {
      listenedRef.current = player.currentTime || 0;
      setCurrentTime(player.currentTime || 0);
      publishMediaPosition(player, trackRef.current?.durationSeconds);
    };
    const metadata = () => {
      const canonical = trackRef.current?.durationSeconds;
      setDuration((current) =>
        canonical && canonical > 0
          ? canonical
          : Number.isFinite(player.duration) && player.duration > 0
            ? player.duration
            : current,
      );
      publishMediaPosition(player, canonical);
    };
    const progress = () =>
      setBuffered(player.buffered.length ? player.buffered.end(player.buffered.length - 1) : 0);
    const ended = () => {
      resetVisuals();
      scrobble(trackRef.current, true);
      window.dispatchEvent(new CustomEvent("echora:playback-state", { detail: false }));
      publishPalette(null);
      if (queueIndexRef.current >= 0 && queueIndexRef.current < queueRef.current.length - 1)
        activateQueueIndexRef.current(queueIndexRef.current + 1);
      else setPlaying(false);
    };
    const paused = () => {
      resetVisuals();
      window.dispatchEvent(new CustomEvent("echora:playback-state", { detail: false }));
      setPlaying(false);
      publishPalette(null);
      if (hasMediaSession()) navigator.mediaSession.playbackState = "paused";
    };
    const waiting = () => {
      resetVisuals();
      setBuffering(true);
    };
    const ready = () => setBuffering(false);
    const started = () => {
      mixer.resume();
      window.dispatchEvent(new CustomEvent("echora:playback-state", { detail: true }));
      setPlaying(true);
      setBuffering(false);
      publishPalette(paletteRef.current);
      if (hasMediaSession()) navigator.mediaSession.playbackState = "playing";
      publishMediaPosition(player, trackRef.current?.durationSeconds);
      const current = trackRef.current;
      if (current?.sourceId && announcedRef.current !== current.sourceId + player.currentSrc) {
        announcedRef.current = current.sourceId + player.currentSrc;
        scrobble(current, false);
      }
    };
    const setAction = (action: MediaSessionAction, handler: MediaSessionActionHandler | null) => {
      try {
        navigator.mediaSession.setActionHandler(action, handler);
      } catch {}
    };
    if (hasMediaSession()) {
      setAction("play", () => {
        startAnalysis();
        player.play().catch(() => setPlaying(false));
      });
      setAction("pause", () => player.pause());
      setAction("previoustrack", () => previousRef.current());
      setAction("nexttrack", () => nextRef.current());
      setAction("seekbackward", (details) =>
        seek(Math.max(0, player.currentTime - (details.seekOffset || 10))),
      );
      setAction("seekforward", (details) => seek(player.currentTime + (details.seekOffset || 10)));
      setAction("seekto", (details) => {
        if (typeof details.seekTime === "number") seek(details.seekTime);
      });
    }
    // Backdrop canvases live outside this provider, so they request the current
    // state instead of relying on a playback event that may have already fired.
    const reportPlaybackState = () =>
      window.dispatchEvent(
        new CustomEvent("echora:playback-state", {
          detail: !player.paused && Boolean(trackRef.current),
        }),
      );
    window.addEventListener("echora:playback-state-request", reportPlaybackState);
    player.addEventListener("timeupdate", time);
    player.addEventListener("durationchange", metadata);
    player.addEventListener("loadedmetadata", metadata);
    player.addEventListener("progress", progress);
    player.addEventListener("ended", ended);
    player.addEventListener("seeking", resetVisuals);
    player.addEventListener("emptied", resetVisuals);
    player.addEventListener("error", resetVisuals);
    player.addEventListener("pause", paused);
    player.addEventListener("play", started);
    player.addEventListener("loadstart", waiting);
    player.addEventListener("waiting", waiting);
    player.addEventListener("canplay", ready);
    player.addEventListener("playing", ready);
    return () => {
      sessionEvents.forEach((event) => player.removeEventListener(event, sessionPlayback));
      window.removeEventListener(SESSION_EXPIRED_EVENT, sessionExpired);
      (
        [
          ["timeupdate", time],
          ["durationchange", metadata],
          ["loadedmetadata", metadata],
          ["progress", progress],
          ["ended", ended],
          ["pause", paused],
          ["play", started],
          ["seeking", resetVisuals],
          ["emptied", resetVisuals],
          ["error", resetVisuals],
          ["loadstart", waiting],
          ["waiting", waiting],
          ["canplay", ready],
          ["playing", ready],
        ] as const
      ).forEach(([event, handler]) => player.removeEventListener(event, handler));
      visualRequest.current?.abort();
      resetVisuals();
      window.removeEventListener("echora:playback-state-request", reportPlaybackState);
      player.pause();
      publishPalette(null);
      if (hasMediaSession()) {
        navigator.mediaSession.metadata = null;
        navigator.mediaSession.playbackState = "none";
        [
          "play",
          "pause",
          "previoustrack",
          "nexttrack",
          "seekbackward",
          "seekforward",
          "seekto",
        ].forEach((action) => setAction(action as MediaSessionAction, null));
      }
      cancelAnimationFrame(analysisFrame.current);
      analysisFrame.current = 0;
      player.remove();
    };
  }, []);

  function startAnalysis() {
    const player = audio.current;
    if (!player || analysisFrame.current) return;
    let clock: ClockAnchor | null = null;
    const analyze = () => {
      analysisFrame.current = requestAnimationFrame(analyze);
      const running = !player.paused && !player.ended && !player.seeking && player.readyState >= 3;
      const smoothed = smoothPlaybackTime(
        clock,
        player.currentTime || 0,
        performance.now(),
        running,
        player.playbackRate || 1,
      );
      clock = smoothed.anchor;
      publishPlaybackTime(smoothed.time);
      if (player.paused || player.ended || player.seeking || player.readyState < 3) {
        if (previousVisualTime.current !== null) resetVisuals();
        return;
      }
      const frame = visualFrameAt(
        visualFeatures.current,
        player.currentTime,
        previousVisualTime.current,
      );
      previousVisualTime.current = frame.active ? player.currentTime : null;
      frame.trackId = trackRef.current?.id ?? null;
      publishVisualFrame(frame);
    };
    analyze();
  }

  function loadLyrics(next: PlayerTrack) {
    const cached = lyricsCacheRef.current.get(next.id);
    if (cached) {
      setLyrics(cached);
      setLyricsLoading(false);
      return;
    }
    setLyrics(null);
    setLyricsLoading(true);
    const generation = lyricsGenerationRef.current.get(next.id) || 0;
    const session = playbackSessionRef.current;
    const resolve = (value: unknown) => {
      // A save or expired session may have invalidated this request in flight.
      if (
        playbackSessionRef.current !== session ||
        (lyricsGenerationRef.current.get(next.id) || 0) !== generation
      )
        return;
      const resolved: PlayerLyrics = value
        ? { ...(value as PlayerLyrics), trackId: next.id }
        : { trackId: next.id, available: false };
      lyricsCacheRef.current.set(next.id, resolved);
      if (trackRef.current?.id === next.id) {
        setLyrics(resolved);
        setLyricsLoading(false);
      }
    };
    fetch(`/api/v1/library/tracks/${next.id}/lyrics`)
      .then((response) => (response.ok ? response.json() : null))
      .then(resolve)
      .catch(() => resolve(null));
  }

  function load(next: PlayerTrack) {
    const player = audio.current;
    if (!player) return;
    visualRequest.current?.abort();
    visualFeatures.current = null;
    resetVisuals();
    setLoadVersion((version) => version + 1);
    const previous = trackRef.current;
    if (previous?.sourceId && previous.sourceId !== next.sourceId) {
      const seconds = listenedRef.current;
      if (seconds >= (previous.durationSeconds || 0) / 2 || seconds >= 240)
        scrobble(previous, true);
    }
    const session = playbackSessionRef.current;
    listenedRef.current = 0;
    announcedRef.current = "";
    paletteRef.current = null;
    paletteTrackRef.current = next.id;
    trackRef.current = next;
    setBuffered(0);
    setBuffering(true);
    publishPalette(null);
    window.dispatchEvent(new CustomEvent("echora:track-change", { detail: next.id }));
    publishMediaMetadata(next);
    loadLyrics(next);
    if (next.coverUrl)
      artworkPalette(sizedPlayerCoverArtUrl(next.coverUrl, 128))
        .then((palette) => {
          if (playbackSessionRef.current !== session || paletteTrackRef.current !== next.id) return;
          paletteRef.current = palette;
          if (!audio.current?.paused) publishPalette(palette);
        })
        .catch(() => {});
    const streamQuality = readPlaybackPreferences().quality;
    setAudioQuality(null);
    fetch(`/api/v1/library/tracks/${next.id}/audio-quality`)
      .then((response) => (response.ok ? response.json() : null))
      .then((value) => {
        if (playbackSessionRef.current === session && value && trackRef.current?.id === next.id)
          setAudioQuality({ ...value, streamQuality });
      })
      .catch(() => {});
    startAnalysis();
    setTrack(next);
    setCurrentTime(0);
    setDuration(next.durationSeconds || 0);
    stemsRef.current = null;
    setStems(null);
    const lookup = stemsSupported() ? lookUpStems(next, session) : null;
    if (lookup && stemsEnabledRef.current) {
      // Start on the stems rather than switching a moment into the song; the previous song
      // stops meanwhile.
      player.pause();
      lookup.then((url) => {
        if (playbackSessionRef.current !== session || trackRef.current?.id !== next.id) return;
        applySource(next, url);
        player.play().catch(() => setPlaying(false));
      });
      return;
    }
    applySource(next, null);
    player.play().catch(() => setPlaying(false));
  }

  /** The current song's stems URL, or null; also records whether the song has stems. */
  function lookUpStems(next: PlayerTrack, session: number): Promise<string | null> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), STEMS_LOOKUP_TIMEOUT_MS);
    return fetch(`/api/v1/library/tracks/${encodeURIComponent(next.id)}/karaoke-stems`, {
      signal: controller.signal,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        const url =
          body?.track_id === next.id && body.available && typeof body.url === "string"
            ? (body.url as string)
            : null;
        if (url && playbackSessionRef.current === session && trackRef.current?.id === next.id) {
          stemsRef.current = { trackId: next.id, url };
          setStems(stemsRef.current);
        }
        return url;
      })
      .catch(() => null)
      .finally(() => clearTimeout(timeout));
  }

  /** Point the player at the stems (a URL) or the original song (null). With ``resume``, keep
   * the position and play state, for a switch in the middle of the song. */
  function applySource(
    next: PlayerTrack,
    stemsUrl: string | null,
    resume?: { time: number; play: boolean },
  ) {
    const player = audio.current;
    const mixer = stemMixer.current;
    if (!player || !mixer) return;
    // Route first: building the graph sets the CORS mode the new source loads with.
    if (stemsUrl || mixer.active) mixer.route(Boolean(stemsUrl));
    player.src = stemsUrl
      ? mediaUrl(stemsUrl)
      : streamUrlForQuality(next.streamUrl, readPlaybackPreferences().quality);
    setStemsPlaying(Boolean(stemsUrl));
    if (!resume) return;
    const session = playbackSessionRef.current;
    player.addEventListener(
      "loadedmetadata",
      () => {
        if (playbackSessionRef.current !== session || trackRef.current?.id !== next.id) return;
        player.currentTime = resume.time;
        if (resume.play) player.play().catch(() => setPlaying(false));
      },
      { once: true },
    );
  }

  function setStemsEnabled(value: boolean) {
    stemsEnabledRef.current = value;
    setStemsEnabledState(value);
    writeStorage(stemsEnabledStorageKey, String(value));
    const player = audio.current;
    const current = trackRef.current;
    const available = stemsRef.current?.trackId === current?.id ? stemsRef.current : null;
    if (!player || !current || !available || value === stemsPlaying) return;
    applySource(current, value ? available.url : null, {
      time: player.currentTime || 0,
      play: !player.paused,
    });
  }

  function setStemsBalance(value: number) {
    const next = Math.min(1, Math.max(-1, value));
    stemMixer.current?.setBalance(next);
    setStemsBalanceState(next);
    writeStorage(stemsBalanceStorageKey, String(next));
  }
  function activateQueueIndex(index: number) {
    const nextTrack = queueRef.current[index];
    if (!nextTrack) return;
    queueIndexRef.current = index;
    setQueueIndex(index);
    load(nextTrack);
  }
  activateQueueIndexRef.current = activateQueueIndex;
  function play(next: PlayerTrack) {
    const player = audio.current;
    if (!player) return;
    if (track?.id !== next.id) {
      queueRef.current = [next];
      queueIndexRef.current = 0;
      setQueue([next]);
      setQueueIndex(0);
      load(next);
    } else {
      startAnalysis();
      player.play().catch(() => setPlaying(false));
    }
  }
  function playQueue(tracks: PlayerTrack[], startIndex = 0) {
    if (!tracks.length) return;
    const index = Math.min(Math.max(startIndex, 0), tracks.length - 1);
    queueRef.current = tracks;
    queueIndexRef.current = index;
    setQueue(tracks);
    setQueueIndex(index);
    load(tracks[index]);
  }
  function playNext(next: PlayerTrack) {
    const current = trackRef.current;
    if (!current || queueIndexRef.current < 0) {
      play(next);
      return;
    }
    if (current.id === next.id) return;
    const queue = queueRef.current.filter((item) => item.id !== next.id);
    const currentIndex = queue.findIndex((item) => item.id === current.id);
    const insertAt = currentIndex < 0 ? queue.length : currentIndex + 1;
    queue.splice(insertAt, 0, next);
    queueIndexRef.current = currentIndex;
    setQueueIndex(currentIndex);
    queueRef.current = queue;
    setQueue(queue);
  }
  function next() {
    if (queueIndexRef.current < queueRef.current.length - 1)
      activateQueueIndex(queueIndexRef.current + 1);
  }
  function previous() {
    const player = audio.current;
    if (!player) return;
    if (player.currentTime > 4 || queueIndexRef.current <= 0) {
      seek(0);
      return;
    }
    activateQueueIndex(queueIndexRef.current - 1);
  }
  nextRef.current = next;
  previousRef.current = previous;
  function clearQueue() {
    queueRef.current = track ? [track] : [];
    queueIndexRef.current = track ? 0 : -1;
    setQueue(queueRef.current);
    setQueueIndex(queueIndexRef.current);
  }
  function toggle() {
    const player = audio.current;
    if (!player || !track) return;
    startAnalysis();
    if (player.paused) player.play();
    else {
      resetVisuals();
      player.pause();
    }
  }
  function seek(seconds: number) {
    const player = audio.current;
    if (!player || !Number.isFinite(seconds)) return;
    const end = Number.isFinite(player.duration)
      ? player.duration
      : (trackRef.current?.durationSeconds ?? Infinity);
    const target = Math.max(0, Math.min(seconds, end));
    resetVisuals();
    player.currentTime = target;
    setCurrentTime(target);
    publishPlaybackTime(target);
    publishMediaPosition(player, trackRef.current?.durationSeconds);
  }
  function toggleMute() {
    const player = audio.current;
    if (!player) return;
    player.muted = !player.muted;
    setMuted(player.muted);
  }
  function setVolume(value: number) {
    const player = audio.current;
    if (!player) return;
    const next = Math.min(1, Math.max(0, value));
    player.volume = next;
    setVolumeState(next);
    // Raising the level is an explicit request to hear audio again.
    if (next > 0 && player.muted) {
      player.muted = false;
      setMuted(false);
    }
    try {
      localStorage.setItem(volumeStorageKey, String(next));
    } catch {
      /* storage may be unavailable */
    }
  }

  useEffect(() => {
    const invalidate = (event: Event) => {
      const trackId = (event as CustomEvent<string>).detail;
      if (typeof trackId !== "string") return;
      lyricsCacheRef.current.delete(trackId);
      lyricsGenerationRef.current.set(trackId, (lyricsGenerationRef.current.get(trackId) || 0) + 1);
      if (trackRef.current?.id === trackId) loadLyrics(trackRef.current);
    };
    const clearTranslations = () => {
      for (const [id, cached] of lyricsCacheRef.current) {
        lyricsCacheRef.current.set(id, { ...cached, translations: [] });
      }
      const current = trackRef.current;
      if (current) {
        lyricsGenerationRef.current.set(
          current.id,
          (lyricsGenerationRef.current.get(current.id) || 0) + 1,
        );
        setLyrics((value) => (value ? { ...value, translations: [] } : value));
      }
    };
    window.addEventListener("echora:lyrics-update", invalidate);
    window.addEventListener("echora:translations-cleared", clearTranslations);
    return () => {
      window.removeEventListener("echora:lyrics-update", invalidate);
      window.removeEventListener("echora:translations-cleared", clearTranslations);
    };
  }, []);

  return (
    <PlayerContext.Provider
      value={{
        track,
        karaokeStems: {
          supported: stemsSupport,
          available: Boolean(stems && stems.trackId === track?.id),
          enabled: stemsEnabled,
          playing: stemsPlaying,
          balance: stemsBalance,
          setEnabled: setStemsEnabled,
          setBalance: setStemsBalance,
        },
        waveform,
        melody,
        audioQuality,
        lyrics,
        lyricsLoading,
        playing,
        buffering,
        currentTime,
        duration,
        buffered,
        muted,
        volume,
        expanded,
        queue,
        queueIndex,
        play,
        playQueue,
        playNext,
        next,
        previous,
        clearQueue,
        toggle,
        seek,
        toggleMute,
        setVolume,
        setExpanded,
      }}
    >
      {children}
      {expanded && track && <FullscreenPlayer />}
    </PlayerContext.Provider>
  );
}

/** Which track is current and whether it is audible, without requiring a provider. */
export function useNowPlaying(): { trackId: string | null; playing: boolean } {
  const value = useContext(PlayerContext);
  return { trackId: value?.track?.id ?? null, playing: Boolean(value?.playing) };
}

export function usePlayer() {
  const value = useContext(PlayerContext);
  if (!value) throw new Error("usePlayer must be used inside PlayerProvider");
  return value;
}
