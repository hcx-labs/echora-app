import { AudioLines } from "lucide-react";

/** Full-page loading state, shown before the session and shell are ready. Matches the sidebar brand. */
export default function AppLoading() {
  return (
    <main className="route-loading" aria-busy="true">
      <span className="route-loading-mark">
        <AudioLines aria-hidden="true" />
        Echora
      </span>
      <span className="route-loading-bar" role="progressbar" aria-label="Loading Echora">
        <span />
      </span>
    </main>
  );
}
