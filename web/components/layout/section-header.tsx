import type { ReactNode } from "react";

// Section title 14px with a 12px hint: the same scale as PaneSection.

export function SectionHeader({ title, description, actions, icon }: { title: string; description?: ReactNode; actions?: ReactNode; icon?: ReactNode }) {
  return <header className="mb-3.5 flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
    <div className="min-w-0">
      <h2 className="flex items-center gap-2 text-sm font-semibold">{icon && <span className="inline-flex text-primary [&_svg]:size-4">{icon}</span>}{title}</h2>
      {description && <p className="mt-0.5 max-w-[68ch] text-xs leading-relaxed text-muted-foreground">{description}</p>}
    </div>
    {actions && <div className="flex items-center gap-3 text-xs text-muted-foreground [&_a]:text-primary [&_a:hover]:underline">{actions}</div>}
  </header>;
}
