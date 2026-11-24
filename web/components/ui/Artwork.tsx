"use client";

import { Disc3 } from "lucide-react";
import LoadingImage from "../media/LoadingImage";
import { useNowPlaying } from "../player/PlayerProvider";
import styles from "./Artwork.module.css";

type Props = { src?: string; alt?: string; className?: string; sizes?: string; trackId?: string };

/**
 * Track or album artwork. Pass `trackId` and the art is accented whenever that
 * track is the one playing, so the current song stands out in every list.
 */
export default function Artwork({ trackId, ...props }: Props) {
  return trackId ? <TrackArtwork trackId={trackId} {...props} /> : <ArtworkFrame {...props} />;
}

// Split out so only artwork that names a track subscribes to player updates.
function TrackArtwork({ trackId, ...props }: Props & { trackId: string }) {
  const now = useNowPlaying();
  const state = now.trackId === trackId ? (now.playing ? "playing" : "paused") : undefined;
  return <ArtworkFrame {...props} state={state} />;
}

function ArtworkFrame({ src, alt = "", className = "", sizes = "48px", state }: Props & { state?: "playing" | "paused" }) {
  return <span className={`${styles.artwork} ${className}`} data-now-playing={state}>
    {src ? <LoadingImage src={src} alt={alt} sizes={sizes} /> : <Disc3 aria-hidden="true" />}
    {state && <span className={styles.nowPlaying} role="img" aria-label={state === "playing" ? "Now playing" : "Paused"}><i /><i /><i /></span>}
  </span>;
}
