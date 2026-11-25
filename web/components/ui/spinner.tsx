import { LoaderCircle } from "lucide-react";
import { cn } from "@/lib/utils";

/** The one loading indicator. Pair it with text, or give it a label when it stands alone. */
export function Spinner({ className, label }: { className?: string; label?: string }) {
  return <LoaderCircle data-slot="spinner" role={label ? "status" : undefined} aria-label={label} aria-hidden={label ? undefined : true} className={cn("size-4 shrink-0 animate-spin text-muted-foreground motion-reduce:animate-[spin_1.6s_linear_infinite]", className)} />;
}

/** Centered loading state for a panel, list or page region. */
export function LoadingState({ label = "Loading…", className }: { label?: string; className?: string }) {
  return <div role="status" className={cn("flex min-h-40 flex-col items-center justify-center gap-3 py-16 text-[13px] text-muted-foreground", className)}><Spinner className="size-5 text-primary" />{label}</div>;
}
