"use client";

import { useId, type ReactNode } from "react";
import { Switch } from "../ui/switch";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "../ui/select";
import { SettingRow } from "./SettingRow";

// 14px title, 12px hint. Pixel sizes, because rem-based text-sm/text-xs follow the 14px root
// and would render smaller than the 13px row labels below. Rows supply their own dividers.
export function SectionHeading({
  title,
  description,
  count,
  id,
}: {
  title: string;
  description?: string;
  count?: number;
  id?: string;
}) {
  return (
    <div id={id} className="flex items-end justify-between gap-3 border-b border-border pb-3">
      <div className="min-w-0">
        <h2 className="text-[14px] font-semibold">{title}</h2>
        {description && (
          <p className="mt-0.5 text-[12px] leading-relaxed text-muted-foreground">{description}</p>
        )}
      </div>
      {count !== undefined && (
        <span
          className="text-[12px] tabular-nums text-muted-foreground"
          aria-label={`${count} entries`}
        >
          {count}
        </span>
      )}
    </div>
  );
}
export function Toggle({
  label,
  checked,
  onChange,
  disabled,
  description,
}: {
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  description?: string;
}) {
  const id = useId();
  return (
    <SettingRow label={label} htmlFor={id} description={description}>
      <Switch id={id} checked={checked} onCheckedChange={onChange} disabled={disabled} />
    </SettingRow>
  );
}
export function Choice({
  value,
  onChange,
  children,
  id,
  disabled,
  "aria-label": ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  id?: string;
  disabled?: boolean;
  "aria-label"?: string;
}) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger id={id} aria-label={ariaLabel} className="w-full min-w-0">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>{children}</SelectContent>
    </Select>
  );
}
export const ChoiceItem = SelectItem;
// One optional feature with a switch per processing location. Syncs run on this
// server or on Modal and do only the features switched on where they run.
export function LocationToggles({
  label,
  description,
  local,
  modal,
  onChange,
  disabled,
  modalUnavailable,
}: {
  label: string;
  description?: string;
  local: boolean;
  modal: boolean;
  onChange: (location: "local" | "modal", next: boolean) => void;
  disabled?: boolean;
  // Shown instead of the Modal switch when the feature cannot run there yet.
  modalUnavailable?: string;
}) {
  const id = useId();
  return (
    <SettingRow label={label} description={description}>
      <div
        className="flex flex-wrap items-center gap-x-5 gap-y-2"
        role="group"
        aria-label={`${label} locations`}
      >
        <label htmlFor={`${id}-local`} className="flex items-center gap-2 text-[13px]">
          <Switch
            id={`${id}-local`}
            checked={local}
            disabled={disabled}
            onCheckedChange={(next) => onChange("local", next)}
          />
          This server
        </label>
        {modalUnavailable ? (
          <span className="text-[12px] text-muted-foreground">{modalUnavailable}</span>
        ) : (
          <label htmlFor={`${id}-modal`} className="flex items-center gap-2 text-[13px]">
            <Switch
              id={`${id}-modal`}
              checked={modal}
              disabled={disabled}
              onCheckedChange={(next) => onChange("modal", next)}
            />
            Modal
          </label>
        )}
      </div>
    </SettingRow>
  );
}
