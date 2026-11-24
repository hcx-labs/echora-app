"use client";

import { AudioLines, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "../ui/button";

export default function Login() {
  const [configured, setConfigured] = useState<boolean | null>(null);
  useEffect(() => {
    fetch("/analysis/auth/oidc/status").then(response => response.json()).then(body => setConfigured(Boolean(body.configured))).catch(() => setConfigured(false));
  }, []);
  return <main className="flex min-h-dvh w-full items-center justify-center bg-nav px-4 py-12">
    <div className="w-full max-w-sm">
      <div className="mb-10 flex items-center gap-2.5 text-xl font-extrabold tracking-tight"><AudioLines className="size-6 text-primary" />Echora</div>
      <h1 className="text-[28px] font-semibold tracking-tight">Welcome back</h1>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">Sign in with your organization&apos;s identity provider to reach your music.</p>
      <Button size="lg" className="mt-8 w-full" type="button" loading={configured === null} disabled={!configured} onClick={() => { window.location.assign(new URL("/analysis/auth/oidc/start", window.location.origin)); }}><ShieldCheck />{configured === null ? "Checking sign-in…" : "Continue with OIDC"}</Button>
      {configured === false && <p role="alert" className="mt-4 text-[13px] text-destructive">OIDC has not been configured for this server.</p>}
      <p className="mt-8 border-t border-border pt-5 text-xs leading-relaxed text-subtle-foreground">Your email address identifies your Echora account.</p>
    </div>
  </main>;
}
