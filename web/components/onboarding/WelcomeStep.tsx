"use client";

import { ArrowRight, Check } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { api, json, type SetupSettings } from "./onboardingApi";
import { StepFrame, type StepNav } from "./StepFrame";

function browserZone() {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

function allZones(current: string[]) {
  let zones: string[] = [];
  try {
    zones = Intl.supportedValuesOf("timeZone");
  } catch {}
  return Array.from(new Set(["UTC", ...current, ...zones])).filter(Boolean);
}

export default function WelcomeStep({
  nav,
  settings,
  plan,
  onSaved,
}: {
  nav: StepNav;
  settings: SetupSettings;
  plan: { title: string; detail: string }[];
  onSaved: (next: Partial<SetupSettings>) => void;
}) {
  const [name, setName] = useState(settings.profile.display_name);
  // A new account starts on UTC; suggest the browser's zone instead.
  const [zone, setZone] = useState(settings.timezone === "UTC" ? browserZone() : settings.timezone);
  const [saving, setSaving] = useState(false);
  const zones = useMemo(() => allZones([settings.timezone, browserZone()]), [settings.timezone]);
  const first = (settings.profile.display_name || settings.profile.email).split(/[\s@]/)[0];

  async function save() {
    const trimmed = name.trim();
    if (!trimmed) return;
    setSaving(true);
    try {
      if (trimmed !== settings.profile.display_name)
        await api("/settings/profile", json("PUT", { display_name: trimmed }));
      if (zone !== settings.timezone)
        await api("/settings/timezone", json("PUT", { timezone: zone }));
      onSaved({
        profile: { ...settings.profile, display_name: trimmed },
        timezone: zone,
      });
      nav.next();
    } catch (reason) {
      toast.error("Could not save your details", {
        description: reason instanceof Error ? reason.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <StepFrame
      nav={nav}
      title={`Welcome to Echora, ${first}`}
      description="Echora listens to your music library and turns it into search, karaoke lyrics, and mixes that follow how songs sound and what they say. Setup takes a few minutes, and everything here can be changed later in Settings."
      primary={
        <Button onClick={save} loading={saving} disabled={!name.trim()}>
          Get started
          <ArrowRight />
        </Button>
      }
    >
      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)]">
        <form
          className="space-y-6"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <div className="space-y-2">
            <Label htmlFor="setup-name">Your name</Label>
            <Input
              id="setup-name"
              value={name}
              maxLength={120}
              autoComplete="name"
              onChange={(event) => setName(event.target.value)}
            />
            <p className="text-[13px] text-muted-foreground">
              Signed in as {settings.profile.email}.
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="setup-timezone">Timezone</Label>
            <Select value={zone} onValueChange={setZone}>
              <SelectTrigger id="setup-timezone" className="w-full max-w-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-80">
                {zones.map((value) => (
                  <SelectItem key={value} value={value}>
                    {value.replaceAll("_", " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[13px] text-muted-foreground">
              Morning and late-night mixes follow this clock.
            </p>
          </div>
          <button type="submit" hidden />
        </form>
        <section aria-labelledby="setup-plan" className="border border-border bg-surface/60 p-5">
          <h2 id="setup-plan" className="text-sm font-semibold">
            What we&apos;ll set up
          </h2>
          <ol className="mt-4 space-y-3.5">
            {plan.map((item, index) => (
              <li key={item.title} className="flex gap-3">
                <span className="grid size-6 shrink-0 place-items-center border border-border-strong text-xs tabular-nums text-muted-foreground">
                  {index === 0 ? <Check className="size-3.5 text-primary" /> : index + 1}
                </span>
                <span>
                  <span className="block text-[13px] font-medium">{item.title}</span>
                  <span className="block text-[13px] leading-relaxed text-muted-foreground">
                    {item.detail}
                  </span>
                </span>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </StepFrame>
  );
}
