import type { PlaybackPreferences } from "./playbackPreferences";

/** Below this, animated backdrops and visualizers are off unless the viewer turns them on
 *  (Settings → Appearance → Visual effects on small screens): phones and small windows trade
 *  them for battery, heat and frame rate. Same query as the CSS rules keyed on
 *  ``html[data-small-screen-effects]``. */
export const SMALL_SCREEN_QUERY = "(max-width: 1199px), (max-height: 719px)";

/** Whether the size rule turns effects off right now. */
export function effectsHiddenBySize(
  query: MediaQueryList,
  preferences: Pick<PlaybackPreferences, "smallScreenEffects">,
): boolean {
  return query.matches && !preferences.smallScreenEffects;
}

/** Mirrors the preference on ``<html>`` so the CSS size rules can follow it. */
export function applySmallScreenEffects(
  preferences: Pick<PlaybackPreferences, "smallScreenEffects">,
) {
  if (typeof document === "undefined") return;
  if (preferences.smallScreenEffects) document.documentElement.dataset.smallScreenEffects = "on";
  else delete document.documentElement.dataset.smallScreenEffects;
}
