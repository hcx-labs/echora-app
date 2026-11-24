import MusicWidget from "./MusicWidget";
export default function CopyrightFooter() {
  return (
    <footer
      data-player-bar
      className="relative z-30 h-[var(--player-height)] shrink-0 border-t border-border bg-nav"
    >
      <MusicWidget />
    </footer>
  );
}
