"use client";
import { Check } from "lucide-react";
import { useState } from "react";
import { ToggleGroup, ToggleGroupItem } from "./toggle-group";
export type FilterTagOption = { value: string; label: string; count?: number };

export function FilterTags({ label, value, options, onChange, emptyText = "No options available", limit = 8 }: { label: string; value: string[]; options: FilterTagOption[]; onChange: (value: string[]) => void; emptyText?: string; limit?: number }) {
  const [expanded, setExpanded] = useState(false);
  const overflow = options.length > limit + 2;
  // Collapsed lists keep selected tags visible even when they fall past the limit.
  const shown = !overflow || expanded ? options : options.filter((option, index) => index < limit || value.includes(option.value));
  return <fieldset className="m-0 min-w-0 border-0 p-0">
    <legend className="sr-only">{label}</legend>
    {options.length ? <>
      <ToggleGroup type="multiple" value={value} onValueChange={onChange} aria-label={label} className="flex w-full flex-wrap justify-start gap-1.5">
        {shown.map(option => <ToggleGroupItem key={option.value} value={option.value} aria-label={`${option.label}${option.count != null ? `, ${option.count} tracks` : ""}`} className="group/tag h-auto min-h-8 max-w-full gap-1.5 rounded-none border border-border-strong bg-surface px-2.5 py-1 text-[13px] font-normal whitespace-normal text-foreground/80 hover:border-[#444] hover:bg-hover hover:text-foreground data-[state=on]:border-primary/60 data-[state=on]:bg-primary/10 data-[state=on]:text-foreground">
          <Check className="hidden size-3.5 text-primary group-data-[state=on]/tag:block" aria-hidden="true" />
          <span>{option.label}</span>
          {option.count != null && <span className="text-xs tabular-nums text-subtle-foreground">{option.count}</span>}
        </ToggleGroupItem>)}
      </ToggleGroup>
      {overflow && <button type="button" onClick={() => setExpanded(value => !value)} className="mt-2 text-xs text-primary hover:underline">{expanded ? "Show fewer" : `Show all ${options.length}`}</button>}
    </> : <p className="text-xs leading-relaxed text-subtle-foreground">{emptyText}</p>}
  </fieldset>;
}
