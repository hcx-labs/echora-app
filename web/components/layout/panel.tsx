import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function PanelHeader({ title, description, actions, compact = false }: { title: string; description?: ReactNode; actions?: ReactNode; compact?: boolean }) {
  return <header className={cn("flex shrink-0 flex-wrap items-start justify-between gap-3 border-b border-border", compact ? "items-center px-4 py-2.5" : "px-5 py-4")}>
    <div className="min-w-0">
      <h2 className={cn("min-w-0 truncate font-semibold", compact ? "text-sm" : "text-[15px]")} title={title}>{title}</h2>
      {description && <p className="mt-1 max-w-[70ch] text-[13px] leading-relaxed text-muted-foreground">{description}</p>}
    </div>
    {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
  </header>;
}
export function Panel({ title, description, actions, children, className }: { title: string; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={cn("min-w-0 border border-border bg-surface/60", className)} aria-label={title}><PanelHeader title={title} description={description} actions={actions} />{children}</section>;
}
export function PanelBody({ children, className }: { children: ReactNode; className?: string }) { return <div className={cn("min-w-0 p-5", className)}>{children}</div>; }
export function PanelFooter({ children }: { children: ReactNode }) { return <footer className="flex flex-wrap items-center justify-end gap-2 border-t border-border px-5 py-3">{children}</footer>; }
