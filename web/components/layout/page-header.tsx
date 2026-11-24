import type { ReactNode } from "react";

export function PageHeader({ title, description, actions, eyebrow }: { title: ReactNode; description?: ReactNode; actions?: ReactNode; eyebrow?: ReactNode }) {
  return <header className="flex shrink-0 flex-wrap items-end justify-between gap-x-6 gap-y-4 px-[var(--gutter)] pt-8 pb-6 max-md:pt-5 max-md:pb-4">
    <div className="min-w-0 flex-1 basis-72">
      {eyebrow && <p className="mb-1.5 text-xs font-medium text-primary">{eyebrow}</p>}
      <h1 className="truncate text-[28px] leading-tight font-semibold tracking-tight max-md:text-[22px]">{title}</h1>
      {description && <p className="mt-1.5 max-w-[68ch] text-[13px] leading-relaxed text-muted-foreground">{description}</p>}
    </div>
    {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
  </header>;
}
