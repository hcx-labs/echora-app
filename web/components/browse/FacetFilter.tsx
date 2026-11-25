import { Check, Search, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Spinner } from "../ui/spinner";

/** A searchable single-choice list. Both instances share one fixed geometry. */
export default function FacetFilter({ label, value, query, items, onQuery, onSelect, loading = false }: { loading?: boolean; label: string; value: string; query: string; items: { name: string; tracks: number }[]; onQuery: (query: string) => void; onSelect: (value: string) => void }) {
  const noun = label.toLowerCase();
  return <div className="min-w-0" role="group" aria-label={label}>
    {value && <div className="mb-2 flex h-9 items-center gap-2 border border-primary/50 bg-primary/10 pr-1 pl-3 text-[13px]"><Check className="size-3.5 text-primary" /><span className="min-w-0 flex-1 truncate">{value}</span><button type="button" aria-label={`Clear ${noun} filter`} onClick={() => onSelect("")} className="grid size-7 place-items-center text-muted-foreground hover:bg-hover hover:text-foreground"><X className="size-3.5" /></button></div>}
    <div className="border border-border bg-surface/50">
      <label className="flex h-9 items-center gap-2 border-b border-border px-3 focus-within:bg-surface">
        <Search className="size-3.5 shrink-0 text-subtle-foreground" />
        <input aria-label={`Search ${noun}`} value={query} onChange={event => onQuery(event.target.value)} placeholder={`Search ${noun}`} className="h-full min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-subtle-foreground" />{loading && <Spinner className="size-3.5" />}
      </label>
      <ul className="h-56 overflow-y-auto py-1 [scrollbar-gutter:stable]">
        {items.map(item => {
          const selected = value === item.name;
          return <li key={item.name}><button type="button" aria-pressed={selected} onClick={() => onSelect(selected ? "" : item.name)} className={cn("grid h-9 w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 text-left text-[13px] transition-colors", selected ? "bg-raised text-primary" : "text-foreground/85 hover:bg-hover hover:text-foreground")}>
            <span className="truncate" title={item.name}>{item.name}</span>
            <span className="min-w-6 text-right text-xs tabular-nums text-subtle-foreground">{item.tracks}</span>
          </button></li>;
        })}
        {!items.length && loading && <li className="flex justify-center py-6"><Spinner label={`Loading ${noun}`} /></li>}{!items.length && !loading && <li className="px-3 py-6 text-center text-xs text-subtle-foreground">{query ? `No ${noun} match “${query}”` : `No ${noun} found`}</li>}
      </ul>
    </div>
  </div>;
}
