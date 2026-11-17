"use client";
import {
  AudioLines,
  ChevronsUpDown,
  Home,
  LibraryBig,
  ListMusic,
  LogOut,
  Orbit,
  Plus,
  RefreshCw,
  Settings,
} from "lucide-react";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { cn } from "@/lib/utils";
import TransitionLink from "./TransitionLink";
import packageInfo from "../../package.json";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";

const links = [
  { href: "/home", label: "Home", Icon: Home },
  { href: "/library", label: "Library", Icon: LibraryBig },
  { href: "/curate", label: "Curations", Icon: ListMusic },
  { href: "/galaxy", label: "Music galaxy", Icon: Orbit },
  { href: "/sync", label: "Sync", Icon: RefreshCw },
  { href: "/settings", label: "Settings", Icon: Settings },
];
import type { SidebarCuration } from "./sidebarCurations";

export default function AppSidebar({
  onboarding = false,
  displayName,
  initials,
  onNavigate,
  mobile = false,
  curations,
}: {
  onboarding?: boolean;
  displayName: string;
  initials: string;
  onNavigate?: () => void;
  mobile?: boolean;
  curations: SidebarCuration[];
}) {
  const pathname = usePathname();
  const [loggingOut, setLoggingOut] = useState(false);
  async function logout() {
    setLoggingOut(true);
    try {
      await fetch("/analysis/auth/logout", { method: "POST" });
    } finally {
      window.location.assign(new URL("/login", window.location.origin));
    }
  }
  const items = onboarding
    ? [{ href: "/setup", label: "Connect library", Icon: LibraryBig }]
    : links;
  const isActive = (href: string) =>
    pathname === href ||
    pathname.startsWith(`${href}/`) ||
    (href === "/library" && pathname.startsWith("/artists/"));

  return (
    <nav aria-label="Main" className="flex h-full min-h-0 flex-col bg-nav">
      <div className="flex h-[72px] shrink-0 items-center px-5 max-md:h-14">
        <TransitionLink
          href={onboarding ? "/setup" : "/home"}
          onClick={onNavigate}
          className="flex items-center gap-2.5 text-xl font-extrabold tracking-tight text-white"
        >
          <AudioLines className="size-5 text-primary" />
          Echora
        </TransitionLink>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-4">
        <ul className="space-y-0.5">
          {items.map(({ href, label, Icon }) => {
            const active = isActive(href);
            return (
              <li key={href}>
                <TransitionLink
                  href={href}
                  onClick={onNavigate}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "relative flex h-10 items-center gap-3 px-3 text-[13px] font-medium transition-colors",
                    active
                      ? "bg-raised text-foreground before:absolute before:inset-y-2.5 before:left-0 before:w-0.5 before:bg-primary"
                      : "text-muted-foreground hover:bg-surface hover:text-foreground",
                  )}
                >
                  <Icon
                    className={cn("size-[18px]", active && "text-primary")}
                    strokeWidth={1.75}
                  />
                  {label}
                </TransitionLink>
              </li>
            );
          })}
        </ul>
        {!onboarding && (
          <section className="mt-8" aria-labelledby="sidebar-curations">
            <div className="mb-2 flex items-center justify-between pr-1 pl-3">
              <h2 id="sidebar-curations" className="text-xs font-medium text-subtle-foreground">
                Your curations
              </h2>
              <TransitionLink
                href="/curate"
                onClick={onNavigate}
                aria-label="Create curation"
                className="grid size-7 place-items-center text-muted-foreground hover:bg-surface hover:text-foreground"
              >
                <Plus className="size-4" />
              </TransitionLink>
            </div>
            {curations.length ? (
              <ul className="space-y-0.5">
                {curations.map((item) => (
                  <li key={item.id}>
                    <TransitionLink
                      href={`/curate/${item.id}`}
                      onClick={onNavigate}
                      aria-current={pathname === `/curate/${item.id}` ? "page" : undefined}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 hover:bg-surface",
                        pathname === `/curate/${item.id}` && "bg-raised text-foreground",
                      )}
                    >
                      <span className="grid size-8 shrink-0 place-items-center border border-border bg-surface text-muted-foreground">
                        <ListMusic className="size-4" />
                      </span>
                      <span className="min-w-0">
                        <span className="block truncate text-[13px] text-foreground">
                          {item.name}
                        </span>
                        <span className="block text-xs text-subtle-foreground">
                          {item.tracks.length} tracks
                        </span>
                      </span>
                    </TransitionLink>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="px-3 text-xs leading-relaxed text-subtle-foreground">
                Saved curations appear here.
              </p>
            )}
          </section>
        )}
      </div>
      <div className="shrink-0 border-t border-border p-3">
        <DropdownMenu>
          <DropdownMenuTrigger className="flex w-full items-center gap-3 p-2 text-left outline-none hover:bg-surface focus-visible:ring-2 focus-visible:ring-ring/60 data-[state=open]:bg-surface">
            <span className="grid size-9 shrink-0 place-items-center border border-border-strong bg-raised text-xs font-semibold text-primary">
              {initials}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-medium">{displayName}</span>
              <span className="block text-xs text-subtle-foreground">Account</span>
            </span>
            <ChevronsUpDown className="size-4 text-subtle-foreground" />
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side={mobile ? "top" : "right"}
            align="end"
            sideOffset={8}
            className="w-56"
          >
            <DropdownMenuLabel className="flex items-center justify-between gap-2 text-xs font-normal text-muted-foreground">
              <span className="flex items-center gap-1.5 font-medium text-foreground">
                <AudioLines className="size-3.5 text-primary" />
                Echora
              </span>
              <span className="tabular-nums">v{packageInfo.version}</span>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            {!onboarding && (
              <DropdownMenuItem asChild>
                <TransitionLink href="/settings" onClick={onNavigate}>
                  <Settings />
                  Settings
                </TransitionLink>
              </DropdownMenuItem>
            )}
            <DropdownMenuItem onSelect={logout} disabled={loggingOut}>
              <LogOut />
              {loggingOut ? "Logging out…" : "Log out"}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </nav>
  );
}
