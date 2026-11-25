"use client";

import { useEffect, useState, type ReactNode } from "react";
import { ScrollArea } from "../ui/scroll-area";
import { Spinner } from "../ui/spinner";
import styles from "./SongInfo.module.css";

type Info = {
  tags: {
    year: number | null;
    genres: string[];
    moods: string[];
    track: number | null;
    disc: number | null;
    album_artists: string[];
    credits: Record<string, string[]>;
    isrc: string | null;
    explicit: boolean;
    comment: string | null;
    play_count: number | null;
    last_played: string | null;
    added: string | null;
    size_bytes: number | null;
    musicbrainz_id: string | null;
  };
  audio: {
    codec: string | null;
    lossless: boolean;
    bit_rate_kbps: number | null;
    bit_depth: number | null;
    sample_rate_hz: number | null;
    channels: number | null;
  };
  analysis: {
    bpm: number | null;
    key: string | null;
    integrated_lufs: number | null;
    loudness_range_lu: number | null;
    true_peak_dbtp: number | null;
  };
  lyrics: {
    available: boolean;
    language: string | null;
    source: string | null;
    synced: boolean;
    karaoke: boolean;
  };
  themes: { description: string; tags: string[] } | null;
};

type Track = {
  id: string;
  title: string;
  artist?: string;
  album?: string;
  durationSeconds?: number;
};

const ROLE_ORDER = [
  "composer",
  "lyricist",
  "arranger",
  "producer",
  "engineer",
  "mixer",
  "performer",
];
const roleLabel = (role: string) => `${role[0].toUpperCase()}${role.slice(1)}`;
const date = (value: string | null) => {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? null
    : parsed.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};
const duration = (seconds?: number) =>
  seconds
    ? `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`
    : null;
const size = (bytes: number | null) =>
  bytes ? `${(bytes / 1024 / 1024).toFixed(bytes > 100 * 1024 * 1024 ? 0 : 1)} MB` : null;
const language = (tag: string | null) => {
  if (!tag) return null;
  try {
    return new Intl.DisplayNames(undefined, { type: "language" }).of(tag) ?? tag;
  } catch {
    return tag;
  }
};
const lyricsSource: Record<string, string> = {
  ai: "AI transcription",
  navidrome: "Navidrome",
  lrclib: "LRCLIB",
};

/** Facts as label / value rows; rows without a value are left out, and so is an empty group. */
function Group({ title, rows }: { title: string; rows: [string, ReactNode][] }) {
  const shown = rows.filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!shown.length) return null;
  return (
    <section className={styles.group}>
      <h3>{title}</h3>
      <dl>
        {shown.map(([label, value]) => (
          <div key={label} className={styles.row}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

/** Song info: a large cover beside everything Echora knows about the track. */
export default function SongInfo({ track, artwork }: { track: Track; artwork: ReactNode }) {
  const [loaded, setLoaded] = useState<{ id: string; info: Info | null } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/v1/library/tracks/${encodeURIComponent(track.id)}/info`, {
      signal: controller.signal,
      credentials: "same-origin",
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((info) => setLoaded({ id: track.id, info }))
      .catch(() => {
        if (!controller.signal.aborted) setLoaded({ id: track.id, info: null });
      });
    return () => controller.abort();
  }, [track.id]);
  const info = loaded?.id === track.id ? loaded.info : null;
  const loading = loaded?.id !== track.id;
  const tags = info?.tags;
  const credits = Object.entries(tags?.credits ?? {}).sort(
    ([a], [b]) =>
      (ROLE_ORDER.indexOf(a) + 1 || 99) - (ROLE_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b),
  );
  const audio = info?.audio;
  const analysis = info?.analysis;
  const chips = [
    ...(tags?.genres ?? []),
    ...(tags?.moods ?? []),
    ...(tags?.explicit ? ["Explicit"] : []),
    ...(audio?.lossless ? ["Lossless"] : []),
  ];
  return (
    <ScrollArea className={styles.scroll}>
      <div className={styles.info}>
        <div className={styles.cover}>{artwork}</div>
        <div className={styles.details}>
          <header className={styles.heading}>
            <h2>{track.title}</h2>
            <p className={styles.artist}>{track.artist || "Unknown artist"}</p>
            <p className={styles.album}>
              {[track.album, tags?.year, duration(track.durationSeconds)]
                .filter(Boolean)
                .join(" · ")}
            </p>
            {chips.length > 0 && (
              <p className={styles.chips}>
                {chips.map((chip) => (
                  <span key={chip}>{chip}</span>
                ))}
              </p>
            )}
          </header>
          {loading ? (
            <p className={styles.status}>
              <Spinner className="size-4 text-current" /> Loading song details…
            </p>
          ) : !info ? (
            <p className={styles.status}>Song details are not available.</p>
          ) : (
            <>
              {info.themes && (
                <section className={styles.about}>
                  <h3>What it&apos;s about</h3>
                  <p>{info.themes.description}</p>
                  {info.themes.tags.length > 0 && (
                    <p className={styles.chips}>
                      {info.themes.tags.map((tag) => (
                        <span key={tag}>{tag}</span>
                      ))}
                    </p>
                  )}
                </section>
              )}
              <div className={styles.groups}>
                <Group
                  title="Credits"
                  rows={[
                    ...(tags!.album_artists.length &&
                    tags!.album_artists.join(", ") !== track.artist
                      ? ([["Album artist", tags!.album_artists.join(", ")]] as [
                          string,
                          ReactNode,
                        ][])
                      : []),
                    ...credits.map(
                      ([role, names]) => [roleLabel(role), names.join(", ")] as [string, ReactNode],
                    ),
                  ]}
                />
                <Group
                  title="Sound"
                  rows={[
                    ["Key", analysis!.key],
                    ["Tempo", analysis!.bpm ? `${Math.round(analysis!.bpm)} BPM` : null],
                    [
                      "Loudness",
                      analysis!.integrated_lufs != null
                        ? `${analysis!.integrated_lufs.toFixed(1)} LUFS`
                        : null,
                    ],
                    [
                      "Dynamic range",
                      analysis!.loudness_range_lu != null
                        ? `${analysis!.loudness_range_lu.toFixed(1)} LU`
                        : null,
                    ],
                    [
                      "True peak",
                      analysis!.true_peak_dbtp != null
                        ? `${analysis!.true_peak_dbtp.toFixed(1)} dBTP`
                        : null,
                    ],
                  ]}
                />
                <Group
                  title="Lyrics"
                  rows={[
                    [
                      "Source",
                      info.lyrics.available
                        ? (lyricsSource[info.lyrics.source ?? ""] ?? info.lyrics.source ?? "Stored")
                        : "None",
                    ],
                    ["Language", info.lyrics.available ? language(info.lyrics.language) : null],
                    [
                      "Timing",
                      info.lyrics.available
                        ? info.lyrics.karaoke
                          ? "Word by word"
                          : info.lyrics.synced
                            ? "Line by line"
                            : "Not timed"
                        : null,
                    ],
                  ]}
                />
                <Group
                  title="File"
                  rows={[
                    ["Format", audio!.codec?.toUpperCase() ?? null],
                    [
                      "Resolution",
                      [
                        audio!.bit_depth ? `${audio!.bit_depth}-bit` : null,
                        audio!.sample_rate_hz
                          ? `${(audio!.sample_rate_hz / 1000).toFixed(1)} kHz`
                          : null,
                      ]
                        .filter(Boolean)
                        .join(" / ") || null,
                    ],
                    ["Bit rate", audio!.bit_rate_kbps ? `${audio!.bit_rate_kbps} kbps` : null],
                    [
                      "Channels",
                      audio!.channels === 2
                        ? "Stereo"
                        : audio!.channels === 1
                          ? "Mono"
                          : audio!.channels,
                    ],
                    ["Size", size(tags!.size_bytes)],
                  ]}
                />
                <Group
                  title="Library"
                  rows={[
                    [
                      "Track",
                      tags!.track
                        ? `${tags!.track}${tags!.disc && tags!.disc > 1 ? ` · disc ${tags!.disc}` : ""}`
                        : null,
                    ],
                    ["Plays", tags!.play_count ?? null],
                    ["Last played", date(tags!.last_played)],
                    ["Added", date(tags!.added)],
                    ["ISRC", tags!.isrc],
                  ]}
                />
              </div>
              {tags!.comment && <p className={styles.comment}>{tags!.comment}</p>}
            </>
          )}
        </div>
      </div>
    </ScrollArea>
  );
}
