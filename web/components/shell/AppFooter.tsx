import { ReactNode } from "react";

export default function AppFooter({
  children,
  marker,
  aside,
}: {
  children: ReactNode;
  marker: ReactNode;
  pinned?: boolean;
  aside?: ReactNode;
}) {
  return (
    <footer className="relative z-30 flex h-[var(--player-height)] shrink-0 items-center gap-4 border-t border-border bg-nav px-[var(--gutter)]">
      <div className="min-w-0 flex-1">{children}</div>
      <div className="shrink-0 text-xs tabular-nums text-muted-foreground empty:hidden">
        {marker}
      </div>
      {aside && <div className="hidden shrink-0 xl:block">{aside}</div>}
    </footer>
  );
}
