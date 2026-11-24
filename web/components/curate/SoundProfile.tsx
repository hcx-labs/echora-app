"use client";

import { Button } from "../ui/button";
import { Switch } from "../ui/switch";
import { Label } from "../ui/label";
import Range from "./Range";
import { useId } from "react";
import styles from "./SoundProfile.module.css";

export const SOUND_PROFILE_AXES = [
  ["pace", "Pace"], ["energy", "Energy"], ["brightness", "Brightness"],
  ["motion", "Motion"], ["vocals", "Vocals"], ["dynamics", "Dynamics"],
] as const;

export type SoundProfileValues = Partial<Record<(typeof SOUND_PROFILE_AXES)[number][0], number>>;
export type SoundProfilePreset = { name: string; values: SoundProfileValues };

export const SOUND_PROFILE_PRESETS: SoundProfilePreset[] = [
  { name: "Still", values: { pace: .25, energy: .2, brightness: .3, motion: .2, vocals: .25, dynamics: .45 } },
  { name: "Warm", values: { pace: .45, energy: .42, brightness: .28, motion: .35, vocals: .55, dynamics: .6 } },
  { name: "Driving", values: { pace: .75, energy: .72, brightness: .58, motion: .7, vocals: .5, dynamics: .5 } },
  { name: "Club", values: { pace: .82, energy: .86, brightness: .7, motion: .8, vocals: .32, dynamics: .38 } },
  { name: "Low vocals", values: { pace: .48, energy: .5, brightness: .48, motion: .45, vocals: .25, dynamics: .58 } },
  { name: "High vocals", values: { pace: .5, energy: .52, brightness: .48, motion: .42, vocals: .9, dynamics: .55 } },
];

export default function SoundProfile({ value, onChange, instrumentalOnly, onInstrumentalOnlyChange }: { value: SoundProfileValues; onChange: (value: SoundProfileValues) => void; instrumentalOnly: boolean; onInstrumentalOnlyChange: (value: boolean) => void }) {
  const id = useId();
  function setAxis(key: (typeof SOUND_PROFILE_AXES)[number][0], next: number) {
    const base = Object.keys(value).length ? value : Object.fromEntries(SOUND_PROFILE_AXES.map(([axis]) => [axis, .5]));
    onChange({ ...base, [key]: next / 100 });
  }
  const active = Object.keys(value).length > 0;
  const point = (index: number, level: number) => {
    const angle = -Math.PI / 2 + index * Math.PI / 3;
    return [180 + Math.cos(angle) * 106 * level, 160 + Math.sin(angle) * 106 * level];
  };
  const points = (level: number) => SOUND_PROFILE_AXES.map((_, index) => point(index, level).join(",")).join(" ");
  return <div className={`${styles.container} grid gap-6`}><div className="flex flex-wrap items-center gap-2"><span className="mr-1 text-xs text-muted-foreground">Presets</span>{SOUND_PROFILE_PRESETS.map(preset => <Button type="button" variant="outline" size="sm" key={preset.name} onClick={() => onChange(preset.values)}>{preset.name}</Button>)}<Button type="button" variant="ghost" size="sm" onClick={() => { onChange({}); onInstrumentalOnlyChange(false); }}>Reset</Button></div>
    <div className={styles.shapeLayout}>
      <figure className={styles.figure}>
        <svg className={styles.chart} viewBox="0 0 360 320" role="img" aria-labelledby={`${id}-title ${id}-description`}>
          <title id={`${id}-title`}>Six-axis sound profile</title>
          <desc id={`${id}-description`}>{active ? SOUND_PROFILE_AXES.map(([key, label]) => `${label}: ${value[key] == null ? "unset" : Math.round(value[key]! * 100) + "%"}`).join("; ") : "No sound shape applied. The neutral outline is a guide, not an active filter."} Adjust the labelled sliders to shape your sound.</desc>
          {[.25, .5, .75, 1].map(level => <polygon key={level} className={styles.ring} points={points(level)} />)}
          {SOUND_PROFILE_AXES.map(([key], index) => { const [x, y] = point(index, 1); return <line key={key} x1="180" y1="160" x2={x} y2={y} className={styles.spoke} />; })}
          <polygon className={active ? styles.shape : styles.neutral} points={SOUND_PROFILE_AXES.map(([key], index) => point(index, value[key] ?? .5).join(",")).join(" ")} />
          {SOUND_PROFILE_AXES.map(([key, label], index) => {
            const [x, y] = point(index, value[key] ?? .5);
            const [labelX, labelY] = point(index, 1.28);
            return <g key={key}><circle cx={x} cy={y} r="4" className={styles.point} /><text x={labelX} y={labelY} textAnchor="middle" dominantBaseline="middle" className={styles.label}>{label}<tspan x={labelX} dy="16">{value[key] == null ? "Unset" : `${Math.round(value[key]! * 100)}%`}</tspan></text></g>;
          })}
        </svg>
        <figcaption className="text-center text-sm text-muted-foreground">{active ? "Targets relative to your library" : "No sound shape applied · adjust a slider or choose a preset"}</figcaption>
      </figure>
      <div className="grid gap-3">{SOUND_PROFILE_AXES.map(([key, label]) => <Range key={key} label={label} value={Math.round((value[key] ?? .5) * 100)} onChange={next => setAxis(key, next)} step={1} unset={value[key] == null} detail={value[key] == null ? "Unset" : `${Math.round(value[key]! * 100)}%`} />)}</div>
    </div>
    <div className="flex items-start gap-3"><Switch id={id} checked={instrumentalOnly} onCheckedChange={onInstrumentalOnlyChange} /><div><Label htmlFor={id}>Instrumental only</Label><p className="mt-1 text-sm text-muted-foreground">Require instrumental tracks. Low vocals allows softer singing.</p></div></div>
  </div>;
}
