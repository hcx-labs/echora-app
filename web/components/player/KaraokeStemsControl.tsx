"use client";

import { AudioWaveform } from "lucide-react";
import { usePlayer } from "./PlayerProvider";
import { Button } from "../ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { Slider } from "../ui/slider";
import { Switch } from "../ui/switch";

// The slider snaps to the full song within this distance of the middle.
const CENTER_SNAP = 0.06;

function balanceLabel(balance: number) {
  if (balance <= -0.995) return "Vocals only";
  if (balance >= 0.995) return "Instrumental only";
  if (Math.abs(balance) < 0.005) return "Full song";
  const percent = Math.round(Math.abs(balance) * 100);
  return balance > 0 ? `Vocals −${percent}%` : `Instrumental −${percent}%`;
}

/** The Stage player's karaoke stems: off streams the original song; on remixes the stems. */
export default function KaraokeStemsControl({
  className,
  activeClassName,
}: {
  className?: string;
  activeClassName?: string;
}) {
  const { karaokeStems: stems } = usePlayer();
  if (!stems.supported || !stems.available) return null;
  const percent = Math.round(stems.balance * 100);
  return (
    <div className={className}>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon-sm"
            className={stems.playing ? activeClassName : undefined}
            title="Karaoke"
            aria-label={stems.playing ? `Karaoke: ${balanceLabel(stems.balance)}` : "Karaoke"}
          >
            <AudioWaveform />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" side="bottom" className="flex w-72 flex-col gap-3">
          <label className="flex items-center justify-between gap-3 text-sm">
            <span>
              <span className="block font-medium">Karaoke</span>
              <span className="block text-xs text-muted-foreground">
                {stems.enabled
                  ? "Separated vocals and instrumental, mixed here"
                  : "Playing the original song"}
              </span>
            </span>
            <Switch checked={stems.enabled} onCheckedChange={stems.setEnabled} />
          </label>
          <div
            className={`flex flex-col gap-1.5 ${stems.enabled ? "" : "pointer-events-none opacity-40"}`}
          >
            <output className="text-center text-xs tabular-nums text-muted-foreground">
              {balanceLabel(stems.balance)}
            </output>
            <Slider
              aria-label="Vocals and instrumental balance"
              aria-valuetext={balanceLabel(stems.balance)}
              min={-100}
              max={100}
              step={1}
              value={percent}
              disabled={!stems.enabled}
              onChange={(event) => {
                const value = Number(event.target.value) / 100;
                stems.setBalance(Math.abs(value) < CENTER_SNAP ? 0 : value);
              }}
              onDoubleClick={() => stems.setBalance(0)}
            />
            <div className="flex justify-between text-[11px] text-muted-foreground">
              <span>Vocals</span>
              <span>Full</span>
              <span>Instrumental</span>
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
