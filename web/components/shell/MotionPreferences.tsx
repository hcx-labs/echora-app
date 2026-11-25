"use client";

import { useEffect } from "react";
import { PlaybackPreferences, readPlaybackPreferences } from "../player/playbackPreferences";

const playbackRate = (speed: PlaybackPreferences["animationSpeed"]) => speed === "slow" ? .65 : speed === "fast" ? 1.45 : 1;

export default function MotionPreferences() {
  useEffect(() => {
    let rate = playbackRate(readPlaybackPreferences().animationSpeed);
    const apply = () => document.getAnimations().forEach(animation => { animation.playbackRate = rate; });
    const receive = (event: Event) => {
      rate = playbackRate((event as CustomEvent<PlaybackPreferences>).detail.animationSpeed);
      apply();
    };
    const started = () => requestAnimationFrame(apply);
    window.addEventListener("echora:playback-preferences", receive);
    document.addEventListener("animationstart", started, true);
    document.addEventListener("transitionrun", started, true);
    apply();
    return () => {
      window.removeEventListener("echora:playback-preferences", receive);
      document.removeEventListener("animationstart", started, true);
      document.removeEventListener("transitionrun", started, true);
    };
  }, []);
  return null;
}
