import type { ReactNode } from "react";

export function EmptyState({ icon, title, description, actions }: { icon?: ReactNode; title: string; description: string; actions?: ReactNode }) {
  return <div className="flex min-h-64 flex-1 flex-col items-center justify-center px-4 py-12 text-center">
    {icon && <span className="mb-5 grid size-12 place-items-center border border-border-strong bg-surface text-primary [&_svg]:size-5">{icon}</span>}
    <h2 className="text-base font-semibold">{title}</h2>
    <p className="mt-2 max-w-sm text-[13px] leading-relaxed text-muted-foreground">{description}</p>
    {actions && <div className="mt-6 flex flex-wrap justify-center gap-2">{actions}</div>}
  </div>;
}
