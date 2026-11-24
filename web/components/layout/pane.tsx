import type { ReactNode, Ref } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * A full-height column with a 72px header, an optional toolbar, a scrolling body
 * and an optional pinned footer. Every multi-column page uses this one geometry.
 */
export function Pane({ title, subtitle, actions, toolbar, footer, children, className, bodyClassName, scrollRef, onScroll, label }: {
  title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; toolbar?: ReactNode; footer?: ReactNode; children: ReactNode;
  className?: string; bodyClassName?: string; scrollRef?: Ref<HTMLDivElement>; onScroll?: () => void; label?: string;
}) {
  return <section aria-label={label ?? (typeof title === "string" ? title : undefined)} className={cn("flex h-full min-h-0 min-w-0 flex-col", className)}>
    <header className="flex min-h-[72px] shrink-0 items-center justify-between gap-3 px-6 py-3 max-md:min-h-14 max-md:px-4">
      <div className="min-w-0"><h2 className="truncate text-lg font-semibold tracking-tight">{title}</h2>{subtitle && <p className="truncate text-xs text-muted-foreground">{subtitle}</p>}</div>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </header>
    {toolbar && <div className="shrink-0 px-6 pb-3 max-md:px-4">{toolbar}</div>}
    <div ref={scrollRef} onScroll={onScroll} className={cn("min-h-0 flex-1 overflow-y-auto px-6 pb-6 max-md:px-4", bodyClassName)}><div className="motion-enter">{children}</div></div>
    {footer && <footer className="flex shrink-0 items-center gap-2 border-t border-border px-6 py-3 max-md:px-4">{footer}</footer>}
  </section>;
}

/**
 * Typography inside panes. Pane title 18px · section title 14px · field label and
 * body 13px · hints, counts and metadata 12px. Use these instead of ad-hoc sizes.
 */
export function PaneSections({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("grid gap-6", className)}>{children}</div>;
}

export function PaneSection({ title, hint, actions, children, open, onOpenChange, className }: {
  title: ReactNode; hint?: ReactNode; actions?: ReactNode; children: ReactNode;
  /** Pass both to make the section collapsible. */
  open?: boolean; onOpenChange?: (open: boolean) => void; className?: string;
}) {
  const collapsible = open !== undefined && onOpenChange !== undefined;
  const heading = <span className="min-w-0"><span className="block text-sm font-semibold">{title}</span>{hint && <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">{hint}</span>}</span>;
  return <section className={cn("min-w-0 border-t border-border pt-6 first:border-t-0 first:pt-0", className)}>
    <div className="flex items-start justify-between gap-3">
      {collapsible ? <button type="button" aria-expanded={open} onClick={() => onOpenChange(!open)} className="flex min-w-0 flex-1 items-start justify-between gap-3 text-left">{heading}<ChevronDown className={cn("mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} /></button> : heading}
      {actions && <div className="flex shrink-0 items-center gap-2 text-xs">{actions}</div>}
    </div>
    {(!collapsible || open) && <div className="mt-3.5 min-w-0">{children}</div>}
  </section>;
}

export function SectionAction({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return <button type="button" onClick={onClick} className="text-xs text-muted-foreground hover:text-foreground">{children}</button>;
}
