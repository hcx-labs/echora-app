"use client";

import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import AppShell from "./AppShell";
import CopyrightFooter from "./CopyrightFooter";

const titles: Record<string, string> = {
  home: "Home", library: "Library", artists: "Artist", curate: "Curations",
  galaxy: "Music galaxy", sync: "Sync", settings: "Settings",
};

export default function AuthenticatedShell({ children }: { children: ReactNode }) {
  const section = usePathname().split("/")[1];
  const title = titles[section] || section.charAt(0).toUpperCase() + section.slice(1);
  return <AppShell title={title} footer={<CopyrightFooter />}>{children}</AppShell>;
}
