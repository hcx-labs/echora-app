"use client";
import { useEffect, useState } from "react";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { FilterTags, type FilterTagOption } from "../ui/filter-tags";
import { type MetadataFilters } from "./metadataFilters";
import { PaneSection, SectionAction } from "../layout/pane";
import { Spinner } from "../ui/spinner";
export { emptyMetadataFilters, type MetadataFilters } from "./metadataFilters";

type Facet = { name: string; tracks: number };
type Options = { language: Facet[]; genre: Facet[]; translation: Facet[] };
const vocals: FilterTagOption[] = [{value:"vocal",label:"Vocal tracks"},{value:"instrumental",label:"Instrumental"},{value:"female",label:"Female vocal evidence"},{value:"male",label:"Male vocal evidence"},{value:"unknown",label:"Not classified"}];
const lyrics: FilterTagOption[] = [{value:"available",label:"Available"},{value:"missing",label:"Missing text"},{value:"ai",label:"AI-generated"}];
const labelOf = (options: FilterTagOption[]) => (value: string) => options.find(option => option.value === value)?.label ?? value;
const facetLabel = (value: string) => value === "unknown" ? "Unknown" : value;
const languageNames = typeof Intl.DisplayNames === "function" ? new Intl.DisplayNames(["en"], { type: "language" }) : null;
export const languageLabel = (code: string) => {
  if (code === "unknown") return "Unknown";
  try { const name = languageNames?.of(code); return name && name !== code ? name : code.toUpperCase(); } catch { return code.toUpperCase(); }
};
export type MetadataTagKey = "vocals" | "lyrics" | "language" | "translation" | "genre";
export const metadataTagGroups: { key: MetadataTagKey; label: string; format: (value: string) => string }[] = [
  { key: "vocals", label: "Vocals", format: labelOf(vocals) },
  { key: "lyrics", label: "Lyrics", format: labelOf(lyrics) },
  { key: "language", label: "Language", format: languageLabel },
  { key: "translation", label: "Translated", format: languageLabel },
  { key: "genre", label: "Genre", format: facetLabel },
];
const facets = (values: Facet[], format = facetLabel): FilterTagOption[] => [...values].sort((a, b) => b.tracks - a.tracks).map(item => ({value:item.name,label:format(item.name),count:item.tracks}));
export default function BrowseMetadataFilters({ value, onChange }: { value: MetadataFilters; onChange: (next: MetadataFilters) => void }) {
  const [options, setOptions] = useState<Options>({ language: [], genre: [], translation: [] });
  const [error, setError] = useState("");
  const [loading,setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    fetch("/analysis/library/filter-options", { signal: abort.signal }).then(async response => {
      if (!response.ok) throw new Error();
      const body = await response.json();
      if (!abort.signal.aborted) { setOptions(body); setError(""); }
    }).catch(() => { if (!abort.signal.aborted) setError("Could not load filter options"); }).finally(() => {if(!abort.signal.aborted)setLoading(false)});
    return () => abort.abort();
  }, [retry]);
  const set = <Key extends keyof MetadataFilters>(key: Key, next: MetadataFilters[Key]) => onChange({ ...value, [key]: next });
  const group = (key: MetadataTagKey, label: string, options: FilterTagOption[], emptyText?: string) => <PaneSection key={key} title={label} actions={value[key].length > 0 ? <SectionAction onClick={() => set(key, [])}>Clear</SectionAction> : loading && !options.length ? <Spinner className="size-3.5" /> : undefined}>
    <FilterTags label={label} options={options} value={value[key]} onChange={next => set(key, next)} emptyText={emptyText} />
  </PaneSection>;
  return <>
    {group("vocals", "Vocals", vocals)}
    {group("lyrics", "Lyrics", lyrics)}
    {group("language", "Lyrics language", facets(options.language, languageLabel), loading ? "Loading languages…" : "No languages indexed")}
    {group("translation", "Translated into", facets(options.translation, languageLabel), loading ? "Loading translations…" : "No translations available")}
    {group("genre", "Genre", facets(options.genre), loading ? "Loading genres…" : "No genres indexed")}
    <PaneSection title="Release year" actions={(value.year_from || value.year_to) && <SectionAction onClick={() => onChange({ ...value, year_from: "", year_to: "" })}>Clear</SectionAction>}>
      <div className="grid grid-cols-2 gap-2"><Input type="number" aria-label="Year from" placeholder="From" min="0" max="9999" value={value.year_from} onChange={e => set("year_from", e.target.value)} /><Input type="number" aria-label="Year to" placeholder="To" min="0" max="9999" value={value.year_to} onChange={e => set("year_to", e.target.value)} /></div>
      {error && <Button variant="outline" size="sm" className="mt-3" type="button" onClick={() => {setLoading(true);setRetry(n => n + 1)}}>{error}. Retry</Button>}
      <p className="mt-4 text-xs leading-relaxed text-subtle-foreground">Tags in a group match any selection; groups combine. Counts cover your whole library.</p>
    </PaneSection>
  </>;
}
