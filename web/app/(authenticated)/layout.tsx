import type { ReactNode } from "react";
import AuthenticatedShell from "@/components/shell/AuthenticatedShell";

export default function AuthenticatedLayout({ children }: { children: ReactNode }) {
  return <AuthenticatedShell>{children}</AuthenticatedShell>;
}
