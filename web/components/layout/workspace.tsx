import type { CSSProperties, ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * A routed page. The optional rail sits beside the content on desktop; pages
 * expose rail content on mobile themselves (sheet, select, or disclosure).
 * Only `WorkspaceBody` scrolls, so headers and toolbars stay in place.
 */
export function Workspace({ sidebar, sidebarWidth = 264, children }: { sidebar?: ReactNode; sidebarWidth?: number; children: ReactNode }) {
  return <div style={{ "--rail-width": `${sidebarWidth}px` } as CSSProperties} className={cn("flex h-full min-h-0 bg-workspace", sidebar && "md:grid md:grid-cols-[min(var(--rail-width),30vw)_minmax(0,1fr)] md:grid-rows-[minmax(0,1fr)]")}>
    {sidebar && <aside className="hidden min-h-0 min-w-0 border-r border-border bg-rail md:block">{sidebar}</aside>}
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">{children}</div>
  </div>;
}

export function WorkspaceBody({ children, className, scrollRef, onScroll }: { children: ReactNode; className?: string; scrollRef?: React.Ref<HTMLDivElement>; onScroll?: () => void }) {
  return <div ref={scrollRef} onScroll={onScroll} className={cn("motion-enter min-h-0 flex-1 overflow-y-auto px-[var(--gutter)] pt-2 pb-12", className)}>{children}</div>;
}
