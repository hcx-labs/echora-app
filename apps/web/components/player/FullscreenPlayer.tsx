"use client";

import {
  ALargeSmall,
  ChevronDown,
  Clapperboard,
  Image as ImageIcon,
  Languages,
  Disc3,
  ListMusic,
  MicVocal,
  Pause,
  Play,
  SkipBack,
  SkipForward,
} from "lucide-react";
import { sizedPlayerCoverArtUrl } from "../media/coverArt";
import LoadingImage from "../media/LoadingImage";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ComponentProps,
  type CSSProperties,
  type ReactNode,
} from "react";
import { audioQualityLabel } from "./audioQuality";
import { usePlayer } from "./PlayerProvider";
import { FULLSCREEN_CLOSE_EVENT } from "./fullscreenEvents";
import styles from "./FullscreenPlayer.module.css";
import motionStyles from "./FullscreenMotion.module.css";
import Artwork from "../ui/Artwork";
import { Spinner } from "../ui/spinner";
import AiLyricsIndicator from "./AiLyricsIndicator";
import LyricsGlow from "./LyricsGlow";
import {
  defaultPlaybackPreferences,
  readPlaybackPreferences,
  type PlaybackPreferences,
} from "./playbackPreferences";
import { Button } from "../ui/button";
import { Tabs, TabsList, TabsTrigger, tabsListVariants } from "../ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Dialog, DialogContent, DialogTitle } from "../ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { ScrollArea } from "../ui/scroll-area";
import { groupSyllablesByWord, lyricWordIsRtl as isRtlText, splitVoices } from "./lyricWords";
import KaraokeLine from "./KaraokeLine";
import LyricsScroller from "./LyricsScroller";
import WaveformSeek from "./WaveformSeek";
import { formatDuration } from "../ui/TrackRow";
import { playbackTimeSnapshot, subscribePlaybackTime } from "./playbackClock";
import LyricTranslationLine from "./LyricTranslationLine";
import { mediaUrl } from "../media/mediaOrigin";
import {
  artworkStyleStorageKey,
  motionArtworkPath,
  motionVideoPath,
  resolveArtworkStyle,
  type ArtworkStyle,
} from "./motionArtwork";

// Phones get their own now-playing layout: full screen, with its own controls.
const compactQuery = "(max-width: 767px)";
function useCompactLayout() {
  return useSyncExternalStore(
    (notify) => {
      const query = window.matchMedia(compactQuery);
      query.addEventListener("change", notify);
      return () => query.removeEventListener("change", notify);
    },
    () => window.matchMedia(compactQuery).matches,
    () => false,
  );
}

function FullscreenMarquee({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [distance, setDistance] = useState(0);

  useEffect(() => {
    const el = ref.current;
    const parent = el?.parentElement;
    const inner = el?.firstElementChild;
    if (!el || !parent || !(inner instanceof HTMLElement)) return;
    const update = () =>
      setDistance(Math.max(0, Math.ceil(inner.offsetWidth - parent.clientWidth)));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(parent);
    observer.observe(inner);
    if (document.fonts?.ready) document.fonts.ready.then(update).catch(() => {});
    return () => observer.disconnect();
  }, [children]);

  return (
    <span
      ref={ref}
      className={styles.fullscreenMarquee}
      data-overflowing={distance > 1 || undefined}
      style={{ "--fullscreen-marquee-offset": `-${distance}px` } as CSSProperties}
    >
      <span>{children}</span>
    </span>
  );
}
type LyricsLine = {
  start_ms: number | null;
  end_ms?: number;
  text: string;
  syllables?: { start_ms: number; end_ms: number; text: string; voice?: number }[];
};
type LyricsTextSize = "small" | "normal" | "large";
const lyricsSizeStorageKey = "echora:lyrics-text-size";
const lyricsSizeClasses: Record<LyricsTextSize, string> = {
  small: styles.lyricsSmall,
  normal: styles.lyricsNormal,
  large: styles.lyricsLarge,
};

export default function FullscreenPlayer() {
  const player = usePlayer();
  const dialogRef = useRef<HTMLElement>(null);
  const compact = useCompactLayout();
  // Phone layout: the cover (or its motion loop), the lyrics, or the queue fill the middle.
  const [mobileView, setMobileView] = useState<"artwork" | "lyrics" | "queue">("artwork");
  const artworkVisible = !compact || mobileView === "artwork";
  // On phones the player takes the whole screen and has its own controls: hide the player bar.
  useEffect(() => {
    if (!compact) return;
    document.body.dataset.playerFullscreen = "";
    return () => {
      delete document.body.dataset.playerFullscreen;
    };
  }, [compact]);
  const [panel, setPanel] = useState<"lyrics" | "queue" | null>(null);
  const [showTranslation, setShowTranslation] = useState(false);
  const [translationLanguage, setTranslationLanguage] = useState("");
  const [karaokeMode, setKaraokeMode] = useState(true);
  const [highlightStyle, setHighlightStyle] = useState(
    defaultPlaybackPreferences.karaokeHighlightStyle,
  );
  useEffect(() => {
    const accept = (value: PlaybackPreferences) =>
      setHighlightStyle(value.karaokeHighlightStyle === "syllable" ? "syllable" : "lava");
    const frame = requestAnimationFrame(() => accept(readPlaybackPreferences()));
    const update = (event: Event) => accept((event as CustomEvent<PlaybackPreferences>).detail);
    window.addEventListener("echora:playback-preferences", update);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("echora:playback-preferences", update);
    };
  }, []);
  const [lyricsTextSize, setLyricsTextSize] = useState<LyricsTextSize>("normal");
  const [artworkStyle, setArtworkStyle] = useState<ArtworkStyle>("motion");
  const [motion, setMotion] = useState<{ trackId: string; src: string } | null>(null);
  const [motionReady, setMotionReady] = useState("");
  const motionVideos = useRef<(HTMLVideoElement | null)[]>([]);
  // Read by the loop below without restarting it, so pause and resume keep the frame.
  const motionPlaying = useRef(player.playing);
  const [closing, setClosing] = useState(false);
  const { setExpanded } = player;
  const close = useCallback(() => {
    if (document.body.classList.contains("echora-fullscreen-player-closing")) return;
    document.body.classList.add("echora-fullscreen-player-closing");
    setClosing(true);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.setTimeout(() => setExpanded(false), reduced ? 0 : 240);
  }, [setExpanded]);
  // The player bar stays live below this view, so its expand button asks us to close with the exit animation.
  useEffect(() => {
    window.addEventListener(FULLSCREEN_CLOSE_EVENT, close);
    return () => window.removeEventListener(FULLSCREEN_CLOSE_EVENT, close);
  }, [close]);
  useEffect(() => {
    document.body.classList.add("echora-fullscreen-player");
    return () => {
      document.body.classList.remove("echora-fullscreen-player");
      document.body.classList.remove("echora-fullscreen-player-closing");
    };
  }, []);
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      setShowTranslation(localStorage.getItem("echora:lyrics-translation") === "true");
      setTranslationLanguage(localStorage.getItem("echora:lyrics-translation-language") || "");
      const stored = localStorage.getItem(lyricsSizeStorageKey);
      if (stored === "small" || stored === "normal" || stored === "large")
        setLyricsTextSize(stored);
      setArtworkStyle(
        resolveArtworkStyle(
          localStorage.getItem(artworkStyleStorageKey),
          window.matchMedia("(prefers-reduced-motion: reduce)").matches,
        ),
      );
    });
    return () => cancelAnimationFrame(frame);
  }, []);
  const trackId = player.track?.id;
  // Motion artwork is looked up per track; most tracks have none, which is a normal answer.
  // The whole clip is loaded into memory before it plays, so looping never goes back to the
  // network: a browser may drop the start of a streamed clip and fetch it again at every loop.
  useEffect(() => {
    if (!trackId) return;
    const controller = new AbortController();
    let objectUrl = "";
    fetch(motionArtworkPath(trackId), { signal: controller.signal, credentials: "same-origin" })
      .then((response) => (response.ok ? response.json() : null))
      .then(async (body) => {
        const path = motionVideoPath(body);
        if (!path) {
          setMotion(null);
          return;
        }
        const video = await fetch(mediaUrl(path), {
          signal: controller.signal,
          credentials: "same-origin",
        });
        if (!video.ok) throw new Error("Motion artwork unavailable");
        objectUrl = URL.createObjectURL(await video.blob());
        setMotion({ trackId, src: objectUrl });
      })
      .catch(() => {
        if (!controller.signal.aborted) setMotion(null);
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [trackId]);
  const currentMotion = motion && motion.trackId === trackId ? motion : null;
  const showMotion = artworkStyle === "motion" && Boolean(currentMotion);
  // Two copies alternate so the opening frames fade over the end of each loop.
  useEffect(() => {
    const videos = motionVideos.current.filter((video): video is HTMLVideoElement =>
      Boolean(video),
    );
    if (!showMotion || videos.length !== 2) return;
    let frame = 0;
    let active = 0;
    let crossing = false;
    const fadeSeconds = 0.65;
    videos.forEach((video, index) => {
      video.pause();
      video.currentTime = 0;
      video.style.opacity = index === 0 ? "1" : "0";
    });
    if (motionPlaying.current) videos[0].play().catch(() => {});
    const draw = () => {
      frame = requestAnimationFrame(draw);
      if (!motionPlaying.current) return;
      const current = videos[active];
      const next = videos[1 - active];
      if (!Number.isFinite(current.duration) || current.duration <= fadeSeconds) return;
      const remaining = current.duration - current.currentTime;
      if (remaining <= fadeSeconds && !crossing) {
        crossing = true;
        next.currentTime = 0;
        next.play().catch(() => {});
      }
      if (crossing) {
        const progress = Math.max(0, Math.min(1, 1 - remaining / fadeSeconds));
        next.style.opacity = String(progress * progress * (3 - 2 * progress));
      }
      if (current.ended || remaining <= 0.03) {
        current.pause();
        current.currentTime = 0;
        current.style.opacity = "0";
        next.style.opacity = "1";
        active = 1 - active;
        crossing = false;
      }
    };
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      videos.forEach((video) => video.pause());
    };
    // The loop's videos remount when the phone layout returns to the artwork view.
  }, [showMotion, currentMotion?.src, artworkVisible]);
  // Pausing the song freezes the loop on its current frame; resuming continues from there.
  useEffect(() => {
    motionPlaying.current = player.playing;
    const videos = motionVideos.current.filter((video): video is HTMLVideoElement =>
      Boolean(video),
    );
    if (!showMotion || videos.length !== 2) return;
    if (!player.playing) {
      videos.forEach((video) => {
        video.dataset.resume = video.paused ? "" : "1";
        video.pause();
      });
      return;
    }
    const resuming = videos.filter((video) => video.dataset.resume === "1");
    // Nothing was playing when paused (for example a fresh loop): start the visible one.
    (resuming.length
      ? resuming
      : videos.filter((video) => video.style.opacity !== "0").slice(0, 1)
    ).forEach((video) => video.play().catch(() => {}));
    videos.forEach((video) => delete video.dataset.resume);
  }, [player.playing, showMotion, currentMotion?.src, artworkVisible]);
  const currentLyrics = player.lyrics?.trackId === player.track?.id ? player.lyrics : null;
  // Each line's lead fragments, and its backing-vocal fragments (voice 2) shown as a second line.
  const lyricFragments = useMemo(
    () =>
      new Map<
        LyricsLine,
        {
          lead: ReturnType<typeof groupSyllablesByWord>[number];
          backing: ReturnType<typeof groupSyllablesByWord>[number];
        }
      >(
        (currentLyrics?.lines || []).map((line) => {
          const { lead, backing } = splitVoices(line.syllables || []);
          return [
            line,
            {
              lead: groupSyllablesByWord(lead).flat(),
              backing: groupSyllablesByWord(backing).flat(),
            },
          ];
        }),
      ),
    [currentLyrics?.lines],
  );
  const translations = currentLyrics?.translations || [];
  const translation =
    translations.find((item) => item.target_language === translationLanguage) || translations[0];
  const translationVisible = showTranslation && Boolean(translation);
  const aiLyrics = currentLyrics?.provenance?.ai_generated === true;
  const karaokeAvailable = Boolean(currentLyrics?.karaoke && currentLyrics.lines?.length);
  const karaokeLines = currentLyrics?.lines || [];
  // Drives the :lang font rules. Urdu shares the Arabic script, so when the
  // language is unknown, letters Arabic does not use identify it.
  const lyricsLanguage =
    currentLyrics?.language ||
    (/[ٹڈڑںےۓ]/.test(currentLyrics?.text || karaokeLines.map((line) => line.text).join(" "))
      ? "ur"
      : undefined);
  const timedLines = (
    (karaokeMode && karaokeAvailable ? karaokeLines : currentLyrics?.provenance?.lines) || []
  ).filter((line) => Number.isFinite(line.start_ms));
  // Subscribing to a derived position (not the time itself) re-renders this view
  // only when the active line changes or finishes, not on every frame.
  const position = useSyncExternalStore(
    subscribePlaybackTime,
    () => linePosition(timedLines, playbackTimeSnapshot() * 1000),
    () => "-1:0",
  );
  const activeLine = Number(position.split(":")[0]);
  const lineSinging = position.endsWith(":1");
  if (!player.track) return null;
  const fullCover = player.track.coverUrl
    ? sizedPlayerCoverArtUrl(player.track.coverUrl, 1200)
    : "";
  function karaokeLine(line: LyricsLine, active: boolean) {
    if (!karaokeMode || !currentLyrics?.karaoke || !line.syllables?.length)
      return line.text || "...";
    const parts = lyricFragments.get(line);
    return (
      <>
        <LiveKaraokeLine
          fragments={parts?.lead || []}
          active={active}
          highlightStyle={highlightStyle}
        />
        {parts?.backing.length ? (
          <span className={styles.backingVoice}>
            <LiveKaraokeLine
              fragments={parts.backing}
              active={active}
              highlightStyle={highlightStyle}
            />
          </span>
        ) : null}
      </>
    );
  }

  /** A line that is not playing: its text, with any backing vocals on their own line as when it plays. */
  function staticLine(line: LyricsLine) {
    const parts = karaokeMode && currentLyrics?.karaoke ? lyricFragments.get(line) : undefined;
    if (!parts?.backing.length) return line.text || "…";
    const text = (fragments: typeof parts.lead) =>
      fragments.map((fragment) => fragment.text).join("");
    return (
      <>
        {text(parts.lead)}
        <span className={styles.backingVoice}>{text(parts.backing)}</span>
      </>
    );
  }

  function toggleTranslation() {
    const next = !showTranslation;
    setShowTranslation(next);
    localStorage.setItem("echora:lyrics-translation", String(next));
  }
  function untimedTranslation() {
    if (!translationVisible || !translation)
      return currentLyrics?.text ? (
        <div
          className={`${styles.untimedTranslations} ${lyricsSizeClasses[lyricsTextSize]}`}
          aria-label="Lyrics"
        >
          {currentLyrics.text.split("\n").map((line, index) => (
            <p key={index} dir="auto">
              {line || "\u00a0"}
            </p>
          ))}
        </div>
      ) : null;
    return (
      <div
        className={`${styles.untimedTranslations} ${lyricsSizeClasses[lyricsTextSize]}`}
        aria-label="Translated lyrics"
      >
        {translation.source_lines.map((text, id) => (
          <p key={id}>
            <span dir="auto">{text}</span>
            <LyricTranslationLine source={text} translation={translation} />
          </p>
        ))}
      </div>
    );
  }
  function translatedLine(line: LyricsLine) {
    return translationVisible && translation ? (
      <LyricTranslationLine translation={translation} source={line.text} />
    ) : null;
  }
  function chooseLyricsTextSize(size: LyricsTextSize) {
    localStorage.setItem(lyricsSizeStorageKey, size);
    setLyricsTextSize(size);
  }
  function chooseArtworkStyle(style: ArtworkStyle) {
    localStorage.setItem(artworkStyleStorageKey, style);
    setArtworkStyle(style);
  }
  const hasLyrics = timedLines.length > 0 || translationVisible || Boolean(currentLyrics?.text);
  // Fullscreen has room for the complete quality breakdown, one tag per fact.
  const qualityParts = player.audioQuality
    ? audioQualityLabel(player.audioQuality).split(" · ")
    : [];
  // Lyrics lead when there are any; otherwise the panel opens on the queue.
  const activePanel = panel ?? (hasLyrics || player.lyricsLoading ? "lyrics" : "queue");
  const upcoming = Math.max(0, player.queue.length - player.queueIndex - 1);
  const lyricsControls = (
    <>
      <AiLyricsIndicator
        key={player.track.id}
        transcribed={aiLyrics}
        translatedLanguages={translations.map((item) => item.target_language)}
        motionArtwork={Boolean(currentMotion)}
      />
      {karaokeAvailable && (
        <Tabs
          value={karaokeMode ? "karaoke" : "synced"}
          onValueChange={(value) => setKaraokeMode(value === "karaoke")}
        >
          <TabsList aria-label="Lyrics timing mode" className={styles.segmented}>
            <TabsTrigger value="karaoke" title="Word-by-word timing">
              <MicVocal />
              <span className={styles.optionLabel}>Karaoke</span>
            </TabsTrigger>
            <TabsTrigger value="synced" title="Line timing">
              <ListMusic />
              <span className={styles.optionLabel}>Lines</span>
            </TabsTrigger>
          </TabsList>
        </Tabs>
      )}
      {hasLyrics && (
        // Built from the same frame and trigger styles as the Karaoke / Lines group beside it.
        <div
          data-slot="tabs-list"
          data-variant="segmented"
          className={`${tabsListVariants({ variant: "segmented" })} ${styles.segmented}`}
        >
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                data-slot="tabs-trigger"
                className={styles.menuTrigger}
                aria-label={`Lyrics text size: ${lyricsTextSize}`}
                title="Lyrics text size"
              >
                <ALargeSmall />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-36">
              <DropdownMenuRadioGroup
                value={lyricsTextSize}
                onValueChange={(value) => chooseLyricsTextSize(value as LyricsTextSize)}
              >
                {(["small", "normal", "large"] as LyricsTextSize[]).map((size) => (
                  <DropdownMenuRadioItem key={size} value={size}>
                    {`${size[0].toUpperCase()}${size.slice(1)}`}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}
      {translation && (
        <div className={styles.group}>
          <Button
            variant="ghost"
            size="icon-sm"
            className={showTranslation ? styles.toggleOn : undefined}
            title="Show translation"
            aria-label="Show translated lyrics"
            aria-pressed={showTranslation}
            onClick={toggleTranslation}
          >
            <Languages />
          </Button>
          {showTranslation && translations.length > 1 && (
            <Select
              value={translation.target_language}
              onValueChange={(value) => {
                setTranslationLanguage(value);
                localStorage.setItem("echora:lyrics-translation-language", value);
              }}
            >
              <SelectTrigger
                aria-label="Translation language"
                size="sm"
                className={styles.translationSelect}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Array.from(new Set(translations.map((item) => item.target_language))).map(
                  (language) => (
                    <SelectItem key={language} value={language}>
                      {language.toUpperCase()}
                    </SelectItem>
                  ),
                )}
              </SelectContent>
            </Select>
          )}
        </div>
      )}
    </>
  );
  // The cover, with its motion loop when there is one; desktop also puts the style switch on it.
  function renderArtwork(styleSwitch: boolean) {
    return (
      <div className={`motion-fade ${styles.art}`} key={`art-${trackId}`}>
        {fullCover ? (
          <LoadingImage
            sizes="(max-width: 767px) 100vw, (max-width: 960px) 64px, 640px"
            src={fullCover}
            alt=""
            priority
          />
        ) : (
          <Disc3 />
        )}
        {showMotion &&
          currentMotion &&
          [0, 1].map((index) => (
            <video
              ref={(video) => {
                motionVideos.current[index] = video;
              }}
              key={`${currentMotion.src}-${index}`}
              className={styles.motionVideo}
              data-ready={motionReady === currentMotion.src || undefined}
              src={currentMotion.src}
              muted
              playsInline
              disablePictureInPicture
              preload="auto"
              aria-hidden="true"
              onCanPlay={() => setMotionReady(currentMotion.src)}
            />
          ))}
        {styleSwitch && currentMotion && (
          <div className={styles.artworkStyle}>
            <Tabs
              value={artworkStyle}
              onValueChange={(value) => chooseArtworkStyle(value as ArtworkStyle)}
            >
              <TabsList aria-label="Artwork style" className={styles.segmented}>
                <TabsTrigger value="still" title="Still cover">
                  <ImageIcon />
                  <span className={styles.optionLabel}>Still</span>
                </TabsTrigger>
                <TabsTrigger value="motion" title="Motion artwork">
                  <Clapperboard />
                  <span className={styles.optionLabel}>Motion</span>
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        )}
      </div>
    );
  }
  const lyricsBody = hasLyrics ? (
    <div
      lang={lyricsLanguage}
      className={`${styles.lyrics} ${karaokeMode && karaokeAvailable ? styles.karaoke : ""} ${lyricsSizeClasses[lyricsTextSize]} ${translationVisible ? styles.bilingual : ""}`}
    >
      {timedLines.length > 0 ? (
        <LyricsScroller
          key={`${player.track.id}-${karaokeMode}`}
          activeIndex={activeLine}
          className={styles.lineList}
          layoutKey={`${lyricsTextSize}-${translationVisible}-${translation?.target_language ?? ""}`}
        >
          {timedLines.map((line, index) => {
            const current = index === activeLine && lineSinging;
            // Backing vocals can still be sung after the next line starts: keep painting the
            // previous line while its timing runs past the current line's start.
            const overlapping =
              index === activeLine - 1 &&
              Number(line.end_ms) > Number(timedLines[activeLine]?.start_ms);
            return (
              <button
                key={`${line.start_ms}-${index}`}
                data-index={index}
                style={
                  {
                    "--distance": Math.min(4, Math.abs(index - Math.max(0, activeLine))),
                  } as CSSProperties
                }
                data-state={current ? "active" : index <= activeLine ? "past" : "next"}
                className={styles.line}
                onClick={() => player.seek(Number(line.start_ms) / 1000)}
              >
                <span dir={isRtlText(line.text) ? "rtl" : "ltr"} className={styles.originalLine}>
                  {current || overlapping ? karaokeLine(line, true) : staticLine(line)}
                </span>
                {translatedLine(line)}
              </button>
            );
          })}
        </LyricsScroller>
      ) : (
        untimedTranslation()
      )}
    </div>
  ) : (
    <div className={styles.panelEmpty}>
      {player.lyricsLoading ? (
        <>
          <Spinner className="size-5 text-current" />
          Looking for lyrics…
        </>
      ) : (
        <>
          <MicVocal className="size-6" />
          <strong>No lyrics for this track</strong>
          <span>Add them from the track menu in your library.</span>
        </>
      )}
    </div>
  );
  const queueBody = (
    <div className={styles.queueBody}>
      <ScrollArea className={styles.queueScroll}>
        <ol className={styles.queueList}>
          {player.queue.map((track, index) => (
            <li key={`${track.id}-${index}`}>
              <button
                type="button"
                className={styles.queueTrack}
                aria-current={index === player.queueIndex ? "true" : undefined}
                onClick={() => player.playQueue(player.queue, index)}
              >
                <span className={styles.queueIndex}>
                  {index === player.queueIndex ? (
                    <Play className="size-3.5" fill="currentColor" />
                  ) : (
                    index + 1
                  )}
                </span>
                <Artwork
                  trackId={track.id}
                  src={track.coverUrl ? sizedPlayerCoverArtUrl(track.coverUrl, 96) : undefined}
                  className={styles.queueArt}
                />
                <span className={styles.queueText}>
                  <strong>{track.title}</strong>
                  <small>
                    {track.artist || "Unknown artist"}
                    {track.album ? ` · ${track.album}` : ""}
                  </small>
                </span>
              </button>
            </li>
          ))}
        </ol>
      </ScrollArea>
      <footer className={styles.queueFooter}>
        <span>{upcoming} upcoming</span>
        <Button variant="ghost" size="sm" onClick={player.clearQueue} disabled={upcoming === 0}>
          Clear upcoming
        </Button>
      </footer>
    </div>
  );
  const hasNext = player.queueIndex >= 0 && player.queueIndex < player.queue.length - 1;
  const toggleMobileView = (view: "lyrics" | "queue") =>
    setMobileView((current) => (current === view ? "artwork" : view));
  // Phones: full screen over the player bar, with its own playback controls.
  const mobileLayout = (
    <div className={`${styles.mobileStage} ${motionStyles.content}`}>
      <header className={styles.mobileHeader}>
        <Button variant="ghost" size="icon" onClick={close} aria-label="Close player">
          <ChevronDown />
        </Button>
        <p className={styles.mobileSource}>
          <small>Playing from</small>
          <strong>{player.track.album || player.track.artist || "Your library"}</strong>
        </p>
        <Button
          variant="ghost"
          size="icon"
          className={mobileView === "queue" ? styles.toggleOn : undefined}
          aria-label={`Up next${upcoming ? `, ${upcoming} tracks` : ""}`}
          aria-pressed={mobileView === "queue"}
          onClick={() => toggleMobileView("queue")}
        >
          <ListMusic />
        </Button>
      </header>
      <section
        className={`motion-fade ${styles.mobileBody}`}
        key={mobileView}
        aria-label={
          mobileView === "artwork" ? "Artwork" : mobileView === "lyrics" ? "Lyrics" : "Up next"
        }
      >
        {mobileView === "artwork" ? (
          <div className={styles.mobileArt}>{renderArtwork(false)}</div>
        ) : mobileView === "lyrics" ? (
          <>
            {hasLyrics && <div className={styles.mobileLyricsControls}>{lyricsControls}</div>}
            {lyricsBody}
          </>
        ) : (
          queueBody
        )}
      </section>
      <section className={styles.mobileTrack} aria-label="Track" key={`track-${player.track.id}`}>
        {mobileView !== "artwork" && (
          <Artwork
            trackId={player.track.id}
            src={
              player.track.coverUrl ? sizedPlayerCoverArtUrl(player.track.coverUrl, 112) : undefined
            }
            className={styles.mobileThumb}
          />
        )}
        <div className={styles.mobileMeta}>
          <h1 className={styles.title}>
            <FullscreenMarquee>{player.track.title}</FullscreenMarquee>
          </h1>
          <p className={styles.artist}>{player.track.artist || "Unknown artist"}</p>
        </div>
        {(qualityParts[0] || aiLyrics) && (
          <span className={styles.tag} title={aiLyrics ? "Lyrics were generated by AI" : undefined}>
            {qualityParts[0] || "AI lyrics"}
          </span>
        )}
      </section>
      <div className={styles.mobileSeek}>
        <WaveformSeek compact />
        <div className={styles.mobileTimes}>
          <time>{formatDuration(player.currentTime)}</time>
          <time>{formatDuration(player.duration)}</time>
        </div>
      </div>
      <div className={styles.mobileControls}>
        <Button
          variant="ghost"
          size="icon"
          disabled={!currentMotion}
          className={showMotion ? styles.toggleOn : undefined}
          aria-label="Motion artwork"
          aria-pressed={showMotion}
          title={currentMotion ? "Motion artwork" : "No motion artwork for this track"}
          onClick={() => {
            chooseArtworkStyle(artworkStyle === "motion" ? "still" : "motion");
            setMobileView("artwork");
          }}
        >
          <Clapperboard />
        </Button>
        <Button variant="ghost" size="icon" onClick={player.previous} aria-label="Previous track">
          <SkipBack fill="currentColor" />
        </Button>
        <Button
          className={styles.mobilePlay}
          onClick={player.toggle}
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
          size="icon"
          onClick={player.next}
          disabled={!hasNext}
          aria-label="Next track"
        >
          <SkipForward fill="currentColor" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className={mobileView === "lyrics" ? styles.toggleOn : undefined}
          aria-label="Lyrics"
          aria-pressed={mobileView === "lyrics"}
          onClick={() => toggleMobileView("lyrics")}
        >
          <MicVocal />
        </Button>
      </div>
    </div>
  );
  // Non-modal: the player bar below stays interactive, so outside clicks must not dismiss the view.
  return (
    <Dialog
      open
      modal={false}
      onOpenChange={(open) => {
        if (!open) close();
      }}
    >
      <DialogContent
        asChild
        showCloseButton={false}
        aria-describedby={undefined}
        onInteractOutside={(event) => event.preventDefault()}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          dialogRef.current?.focus({ preventScroll: true });
        }}
      >
        <main
          ref={dialogRef}
          tabIndex={-1}
          className={`${styles.player} ${compact ? styles.mobile : ""} ${motionStyles.surface} ${closing ? `${styles.closing} ${motionStyles.closing}` : ""}`}
          aria-label="Now playing"
        >
          <DialogTitle className="sr-only">Now playing</DialogTitle>
          {karaokeMode && karaokeAvailable && (
            <LyricsGlow container={dialogRef} playing={player.playing} />
          )}
          {compact ? (
            mobileLayout
          ) : (
            <div className={`${styles.stage} ${motionStyles.content}`}>
              <section className={styles.nowPlaying} aria-label="Track">
                {renderArtwork(true)}
                <div className={`motion-fade ${styles.meta}`} key={`meta-${player.track.id}`}>
                  <h1 className={styles.title}>
                    <FullscreenMarquee>{player.track.title}</FullscreenMarquee>
                  </h1>
                  <p className={styles.artist}>{player.track.artist || "Unknown artist"}</p>
                  <p className={styles.album}>{player.track.album || "Unknown album"}</p>
                  {(qualityParts.length > 0 || aiLyrics) && (
                    <p className={styles.tags} aria-label="Audio quality">
                      {qualityParts.map((part) => (
                        <span key={part} className={styles.tag}>
                          {part}
                        </span>
                      ))}
                      {aiLyrics && (
                        <span
                          className={styles.tag}
                          title="Lyrics were generated by AI and may be inaccurate"
                        >
                          AI lyrics
                        </span>
                      )}
                    </p>
                  )}
                </div>
              </section>

              <section className={styles.panel} aria-label="Lyrics and queue">
                <header className={styles.panelHeader}>
                  <Tabs
                    value={activePanel}
                    onValueChange={(value) => setPanel(value as "lyrics" | "queue")}
                    className="gap-0"
                  >
                    <TabsList variant="line" aria-label="Panel" className={styles.panelTabs}>
                      <TabsTrigger value="lyrics">Lyrics</TabsTrigger>
                      <TabsTrigger value="queue">
                        Up next{upcoming > 0 && <span className={styles.count}>{upcoming}</span>}
                      </TabsTrigger>
                    </TabsList>
                  </Tabs>
                  {activePanel === "lyrics" && (
                    <div className={styles.panelControls}>{lyricsControls}</div>
                  )}
                </header>
                <div className={`motion-fade ${styles.panelBody}`} key={activePanel}>
                  {activePanel === "lyrics" ? lyricsBody : queueBody}
                </div>
              </section>
            </div>
          )}
        </main>
      </DialogContent>
    </Dialog>
  );
}

/**
 * "index:singing" for the given time. A line stops singing once its own timing
 * ends, even if the next line has not started (instrumental gaps); the scroller
 * still anchors on the index.
 */
function linePosition(lines: LyricsLine[], nowMs: number): string {
  const index = lines.reduce(
    (active, line, i) => (Number(line.start_ms) <= nowMs ? i : active),
    -1,
  );
  const line = lines[index];
  if (!line) return "-1:0";
  const ends = [line.end_ms, ...(line.syllables || []).map((syllable) => syllable.end_ms)].filter(
    (value): value is number => Number.isFinite(value),
  );
  const end = ends.length ? Math.max(...ends) : Infinity;
  return `${index}:${nowMs < end ? 1 : 0}`;
}

/** The only lyric element that follows the clock every frame. */
function LiveKaraokeLine(props: Omit<ComponentProps<typeof KaraokeLine>, "now">) {
  const now = useSyncExternalStore(subscribePlaybackTime, playbackTimeSnapshot, () => 0) * 1000;
  return <KaraokeLine {...props} now={now} />;
}
