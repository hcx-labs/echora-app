"use client";
import { useId } from "react";
import { Slider } from "../ui/slider";
export default function Range({ label, value, onChange, detail, max = 100, step = 5, unset = false }: { label: string; value: number; onChange: (value: number) => void; detail?: string; max?: number; step?: number; unset?: boolean }) {
  const id = useId();
  return <div className="grid gap-1.5">
    <div className="flex flex-wrap items-baseline justify-between gap-2"><label htmlFor={id} className="text-[13px] font-medium">{label}</label><output htmlFor={id} className="text-xs tabular-nums text-muted-foreground">{detail || value}</output></div>
    <Slider id={id} min={0} max={max} step={step} value={value} unset={unset} onChange={event => onChange(Number(event.target.value))} />
  </div>;
}
