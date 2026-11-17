import { ArrowLeft, ArrowRight } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "../ui/button";

export type StepNav = {
  index: number;
  total: number;
  optional: boolean;
  back: (() => void) | null;
  next: () => void;
};

/** One onboarding step: heading, content, and a bar pinned to the bottom with Back and the
 * step's main action. Settings inside a step save as they change, so optional steps need no
 * separate Skip. */
export function StepFrame({
  nav,
  title,
  description,
  children,
  primary,
  aside,
}: {
  nav: StepNav;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  /** The main action; defaults to Continue. */
  primary?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="flex flex-1 flex-col">
      <div className="mx-auto w-full max-w-[880px] flex-1 px-[var(--gutter)] pt-10 pb-12 max-md:pt-6">
        <p className="mb-2 text-xs font-medium tracking-wide text-primary">
          Step {nav.index + 1} of {nav.total}
          {nav.optional && (
            <span className="text-muted-foreground"> · Optional, also in Settings later</span>
          )}
        </p>
        <h1 className="text-[30px] leading-tight font-semibold tracking-tight max-md:text-[24px]">
          {title}
        </h1>
        {description && (
          <p className="mt-2 max-w-[64ch] text-[14px] leading-relaxed text-muted-foreground">
            {description}
          </p>
        )}
        <div className="motion-enter mt-8">{children}</div>
      </div>
      <div className="sticky bottom-0 z-10 border-t border-border bg-nav/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-[880px] items-center gap-2 px-[var(--gutter)] py-3">
          {nav.back ? (
            <Button variant="ghost" onClick={nav.back}>
              <ArrowLeft />
              <span className="max-md:sr-only">Back</span>
            </Button>
          ) : (
            <span />
          )}
          <div className="ml-auto flex items-center gap-2">
            {aside}
            {primary ?? (
              <Button onClick={nav.next}>
                Continue
                <ArrowRight />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
