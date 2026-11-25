"use client";

import Image from "next/image";
import { useState } from "react";
import styles from "./LoadingImage.module.css";

const loadedArtwork = new Set<string>();

export default function LoadingImage({ src, alt = "", sizes, priority = false, className = "" }: { src: string; alt?: string; sizes: string; priority?: boolean; className?: string }) {
  const [loadedSrc, setLoadedSrc] = useState(() => loadedArtwork.has(src) ? src : "");
  const loaded = loadedSrc === src || loadedArtwork.has(src);
  function markLoaded() {
    loadedArtwork.add(src);
    setLoadedSrc(src);
  }
  return <span className={`${styles.frame} ${className}`}>
    {!loaded && <i className={styles.spinner} aria-label="Loading artwork" />}
    <Image unoptimized fill sizes={sizes} priority={priority} src={src} alt={alt} className={loaded ? styles.loaded : styles.loading} onLoad={markLoaded} />
  </span>;
}
