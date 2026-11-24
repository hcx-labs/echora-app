import type { ComponentProps } from "react";
import { Slider } from "../ui/slider";
// Native range semantics and keyboard support, without text-input chrome.
export function RangeControl(props: Omit<ComponentProps<"input">, "type" | "className">) {
  return <Slider {...props} />;
}
