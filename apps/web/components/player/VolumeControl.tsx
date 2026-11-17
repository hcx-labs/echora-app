"use client";

import { useEffect, useRef, useState } from "react";
import { Volume, Volume1, Volume2, VolumeX } from "lucide-react";
import { usePlayer } from "./PlayerProvider";
import { Button } from "../ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";

/** One mouse-wheel notch (100 px in Chromium, 3 lines elsewhere) moves the volume this much. */
const STEP_PER_NOTCH = 0.05;
const PIXELS_PER_NOTCH = 100;

/** Scrolling over `element` turns the volume up (wheel up) or down, instead of the page.
 *  A native listener, because React's wheel handler is passive and cannot stop the scroll.
 *  Trackpads move it smoothly in proportion to the distance scrolled. */
function useWheelVolume(
  element: HTMLElement | null,
  level: number,
  setVolume: (value: number) => void,
) {
  const current = useRef({ level, setVolume });
  useEffect(() => {
    current.current = { level, setVolume };
  }, [level, setVolume]);
  useEffect(() => {
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
      if (!delta) return;
      event.preventDefault();
      const pixels =
        event.deltaMode === 1
          ? (delta * PIXELS_PER_NOTCH) / 3
          : event.deltaMode === 2
            ? delta * PIXELS_PER_NOTCH * 10
            : delta;
      const next =
        Math.round((current.current.level - (pixels / PIXELS_PER_NOTCH) * STEP_PER_NOTCH) * 100) /
        100;
      current.current.level = Math.min(1, Math.max(0, next));
      current.current.setVolume(current.current.level);
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [element]);
}

/** Volume icon that opens a vertical level slider above it; the icon reflects the current level.
 *  Scrolling over the icon or the open slider adjusts the volume. */
export default function VolumeControl({
  className,
  side = "top",
}: {
  className?: string;
  side?: "top" | "bottom";
}) {
  const player = usePlayer();
  const level = player.muted ? 0 : player.volume;
  const Icon = level === 0 ? VolumeX : level < 0.34 ? Volume : level < 0.67 ? Volume1 : Volume2;
  const percent = Math.round(level * 100);
  const [trigger, setTrigger] = useState<HTMLButtonElement | null>(null);
  const [panel, setPanel] = useState<HTMLDivElement | null>(null);
  const enabled = Boolean(player.track);
  useWheelVolume(enabled ? trigger : null, level, player.setVolume);
  useWheelVolume(enabled ? panel : null, level, player.setVolume);
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          ref={setTrigger}
          variant="ghost"
          size="icon"
          className={className}
          disabled={!player.track}
          aria-label={`Volume ${percent}%`}
        >
          <Icon className="size-[18px]" />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        ref={setPanel}
        side={side}
        align="center"
        className="flex w-12 flex-col items-center gap-2 px-0 py-3"
      >
        <output className="text-xs tabular-nums text-muted-foreground">{percent}</output>
        {/* A rotated horizontal range: consistent across engines, unlike vertical writing-mode ranges. */}
        <div className="relative h-32 w-5">
          <Slider
            aria-label="Volume"
            aria-orientation="vertical"
            min={0}
            max={100}
            step={1}
            value={percent}
            onChange={(event) => player.setVolume(Number(event.target.value) / 100)}
            className="absolute top-1/2 left-1/2"
            style={{ width: 128, transform: "translate(-50%, -50%) rotate(-90deg)" }}
          />
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={player.toggleMute}
          aria-label={player.muted ? "Unmute" : "Mute"}
          aria-pressed={player.muted}
        >
          <Icon />
        </Button>
      </PopoverContent>
    </Popover>
  );
}
