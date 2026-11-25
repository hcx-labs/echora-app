import type { ReactNode } from "react";
import { CircleAlert, CircleCheck, Info, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Inline banner for consequences the reader should not miss: what will (not)
 * apply, what is irreversible, what costs money. Plain field help stays a hint.
 */
const tones = {
  info: { Icon: Info, className: "border-primary/30 bg-primary/[.06] [&>svg]:text-primary" },
  warning: { Icon: TriangleAlert, className: "border-warning/35 bg-warning/[.07] [&>svg]:text-warning" },
  error: { Icon: CircleAlert, className: "border-destructive/35 bg-destructive/[.07] [&>svg]:text-destructive" },
  success: { Icon: CircleCheck, className: "border-success/35 bg-success/[.07] [&>svg]:text-success" },
} as const;

export type NoticeTone = keyof typeof tones;

export function Notice({ tone = "info", title, children, action, className }: { tone?: NoticeTone; title?: ReactNode; children?: ReactNode; action?: ReactNode; className?: string }) {
  const { Icon, className: toneClass } = tones[tone];
  return <div role={tone === "error" || tone === "warning" ? "alert" : "status"} data-slot="notice" className={cn("motion-fade flex items-start gap-2.5 border px-3 py-2.5 text-[13px] leading-relaxed", toneClass, className)}>
    <Icon className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
    <div className="min-w-0 flex-1">
      {title && <p className="font-medium text-foreground">{title}</p>}
      {children && <div className={cn("text-muted-foreground", title && "mt-0.5")}>{children}</div>}
    </div>
    {action && <div className="shrink-0 self-center">{action}</div>}
  </div>;
}
