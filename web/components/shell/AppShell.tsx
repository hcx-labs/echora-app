"use client";

import { Menu } from "lucide-react";
import { useRouter } from "next/navigation";
import { ReactNode, useEffect, useState } from "react";
import AppSidebar from "./AppSidebar";
import { useSidebarCurations } from "./sidebarCurations";
import { Button } from "../ui/button";
import AppLoading from "./AppLoading";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "../ui/sheet";

import {
  cacheUser,
  getCachedUser,
  invalidateSessionUser,
  sessionGeneration,
  SESSION_EXPIRED_EVENT,
  type ShellUser,
} from "../session/sessionUser";

export default function AppShell({
  title,
  footer,
  children,
  onboarding = false,
}: {
  title: string;
  footer: ReactNode;
  children: ReactNode;
  onboarding?: boolean;
}) {
  const router = useRouter();
  const curations = useSidebarCurations(onboarding);
  const [user, setUser] = useState<ShellUser | null>(getCachedUser);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    let active = true;
    const generation = sessionGeneration();
    const accept = (value: ShellUser) => {
      if (!active || generation !== sessionGeneration()) return;
      cacheUser(value);
      if (onboarding && value.onboarding_complete) {
        router.replace("/home");
        return;
      }
      if (!onboarding && !value.onboarding_complete) {
        router.replace("/setup");
        return;
      }
      setUser(value);
    };
    const update = (event: Event) => {
      const detail = (event as CustomEvent<Partial<ShellUser>>).detail;
      const cachedUser = getCachedUser();
      if (cachedUser) accept({ ...cachedUser, ...detail });
    };
    const expired = () => {
      setUser(null);
      router.replace("/login");
    };
    window.addEventListener(SESSION_EXPIRED_EVENT, expired);
    window.addEventListener("echora:user-update", update);
    const cachedUser = getCachedUser();
    if (cachedUser) {
      if (onboarding && cachedUser.onboarding_complete) router.replace("/home");
      else if (!onboarding && !cachedUser.onboarding_complete) router.replace("/setup");
    } else
      fetch("/api/v1/auth/me")
        .then((response) => {
          if (!active || generation !== sessionGeneration()) return null;
          if (response.status === 401) {
            invalidateSessionUser();
            return null;
          }
          if (!response.ok) throw new Error("Could not verify session");
          return response.json();
        })
        .then((value) => value && accept(value))
        .catch(() => {
          if (active && generation === sessionGeneration()) router.replace("/login");
        });
    return () => {
      active = false;
      window.removeEventListener("echora:user-update", update);
      window.removeEventListener(SESSION_EXPIRED_EVENT, expired);
    };
  }, [onboarding, router]);

  if (!user) return <AppLoading />;
  const displayName = user.display_name || user.username;
  const initials =
    displayName
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0])
      .join("")
      .toUpperCase() || "EC";

  return (
    <div
      aria-label={title}
      className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)_auto] bg-nav md:grid-cols-[var(--sidebar-width)_minmax(0,1fr)]"
    >
      <div className="hidden min-h-0 border-r border-border md:block">
        <AppSidebar
          curations={curations}
          onboarding={onboarding}
          displayName={displayName}
          initials={initials}
        />
      </div>
      <div className="flex min-h-0 min-w-0 flex-col">
        <div className="flex h-14 shrink-0 items-center gap-2 border-b border-border bg-nav px-2 md:hidden">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Open navigation"
            onClick={() => setMenuOpen(true)}
          >
            <Menu />
          </Button>
          <span className="truncate text-[15px] font-semibold">{title}</span>
        </div>
        <main className="motion-fade min-h-0 flex-1 overflow-hidden bg-workspace">{children}</main>
      </div>
      <div className="md:col-span-2">{footer}</div>
      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="w-[280px] gap-0 p-0 [&>button]:hidden">
          <SheetHeader className="sr-only">
            <SheetTitle>Navigation</SheetTitle>
            <SheetDescription>Main navigation and account menu.</SheetDescription>
          </SheetHeader>
          <AppSidebar
            curations={curations}
            mobile
            onboarding={onboarding}
            displayName={displayName}
            initials={initials}
            onNavigate={() => setMenuOpen(false)}
          />
        </SheetContent>
      </Sheet>
    </div>
  );
}
