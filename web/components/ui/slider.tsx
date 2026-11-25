import type { ComponentProps, CSSProperties } from "react";
import { cn } from "@/lib/utils";

/**
 * A quiet native range: hairline track, neutral fill, small square thumb.
 * `unset` dims it to show the value is a placeholder, not an active choice.
 */
export function Slider({ className, unset = false, style, ...props }: Omit<ComponentProps<"input">, "type"> & { unset?: boolean }) {
  const min = Number(props.min ?? 0), max = Number(props.max ?? 100), value = Number(props.value ?? props.defaultValue ?? min);
  const fill = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return <input type="range" data-slot="slider" data-unset={unset || undefined} className={cn("echora-range", className)} style={{ "--fill": `${fill}%`, ...style } as CSSProperties} {...props} />;
}
