import type { ReactNode } from "react";
import { Pause, Play } from "lucide-react";
import Artwork from "./Artwork";
import styles from "./TrackRow.module.css";
export const formatDuration = (seconds: number) => `${Math.floor(Math.max(0, seconds) / 60)}:${String(Math.floor(Math.max(0, seconds) % 60)).padStart(2, "0")}`;
export default function TrackRow({ title, artist, album, coverUrl, duration, index, active = false, playing = false, disabled = false, onPlay, actions, detail, children }: { title: string; artist?: string; album?: string; coverUrl?: string; duration: number; index?: number; active?: boolean; playing?: boolean; disabled?: boolean; onPlay: () => void; actions?: ReactNode; detail?: ReactNode; children?: ReactNode }) {
  return <article className={`${styles.row} ${active ? styles.active : ""}`}>
    <button type="button" className={styles.playTrack} onClick={onPlay} disabled={disabled} aria-label={`${active && playing ? "Pause" : "Play"} ${title}`}>
      <span className={styles.index}>{active && playing ? <Pause size={13} /> : index != null ? String(index + 1).padStart(2, "0") : <Play size={13} />}</span>
      <Artwork src={coverUrl} className={styles.art} />
      <span className={styles.identity}><strong>{title}</strong><small>{artist || "Unknown artist"}</small>{detail && <span className={styles.detail}>{detail}</span>}</span>
      <span className={styles.album}>{album || "Unknown album"}</span><time>{formatDuration(duration)}</time>
    </button>
    {actions && <div className={styles.actions}>{actions}</div>}
    {children && <div className={styles.extra}>{children}</div>}
  </article>;
}
