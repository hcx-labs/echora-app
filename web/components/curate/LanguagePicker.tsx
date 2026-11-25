"use client";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";

type Props = { value: string; onChange: (value: string) => void; options: [string, string][]; ariaLabel?: string; id?: string };
export default function LanguagePicker({ value, onChange, options, ariaLabel, id }: Props) {
  return <Select value={value || "any"} onValueChange={next => onChange(next === "any" ? "" : next)}><SelectTrigger id={id} className="w-full" aria-label={ariaLabel}><SelectValue /></SelectTrigger><SelectContent>{options.map(([code, label]) => <SelectItem key={code || "any"} value={code || "any"}>{label}</SelectItem>)}</SelectContent></Select>;
}
