"use client";

import { Search, X } from "lucide-react";
import { useEffect, useState, useId } from "react";
import { Button } from "../ui/button";
import { Notice } from "../ui/notice";
import { Input } from "../ui/input";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "../ui/dialog";
import { Label } from "../ui/label";
import { LoadingState } from "../ui/spinner";
import { ScrollArea } from "../ui/scroll-area";

export type ReferenceTrack = { id: string; title: string; artist?: string; album?: string };

export default function TrackReferencePicker({ label, value, onChange, single = false }: { label: string; value: ReferenceTrack[]; onChange: (tracks: ReferenceTrack[]) => void; single?: boolean }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ReferenceTrack[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!open || query.trim().length < 2) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setLoading(true); setError(""); setResults([]);
      try {
        const params = new URLSearchParams({ q: query.trim(), limit: "8", offset: "0", sort_by: "name" });
        const response = await fetch(`/analysis/library/tracks?${params}`, { signal: controller.signal });
        if (!response.ok) throw new Error("Track search failed. Try again.");
        const body = await response.json();
        if (!controller.signal.aborted) setResults((body.tracks || []).filter((track: ReferenceTrack) => !value.some(item => item.id === track.id)));
      } catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Search failed"); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    }, 220);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [open, query, value]);
  function changeQuery(next: string) { setQuery(next); setResults([]); setError(""); setLoading(next.trim().length >= 2); }
  return <div className="grid content-start gap-2.5"><div className="text-[13px] font-medium">{label}</div>
    {value.length > 0 && <ol className="grid gap-1.5">{value.map((track, index) => <li key={track.id} className="flex h-11 min-w-0 items-center gap-2 border border-border-strong bg-raised pr-1 pl-3 text-[13px]">{!single && value.length > 1 && <span className="w-4 text-xs tabular-nums text-subtle-foreground">{index + 1}</span>}<span className="min-w-0 flex-1"><span className="block truncate font-medium">{track.title}</span>{track.artist && <span className="block truncate text-xs text-muted-foreground">{track.artist}</span>}</span><Button type="button" variant="ghost" size="icon-sm" aria-label={`Remove ${track.title}`} onClick={() => onChange(value.filter(item => item.id !== track.id))}><X /></Button></li>)}</ol>}
    <Dialog open={open} onOpenChange={next => { setOpen(next); changeQuery(""); }}><DialogTrigger asChild><Button variant="outline" className="justify-start border-dashed"><Search />{single && value.length ? "Replace track" : "Add track"}</Button></DialogTrigger><DialogContent><DialogHeader><DialogTitle>{label}</DialogTitle><DialogDescription>Search your library{single ? " for one waypoint." : ". Tracks are added in selection order."}</DialogDescription></DialogHeader><Label htmlFor={id}>Search title or artist</Label><Input id={id} autoFocus value={query} onChange={event => changeQuery(event.target.value)} placeholder="Search title or artist" /><ScrollArea className="h-72"><div className="space-y-1">{query.trim().length < 2 ? <p className="p-4 text-sm text-muted-foreground">Enter at least two characters.</p> : loading ? <LoadingState label="Searching your library…" className="min-h-0 py-10" /> : error ? <Notice tone="error">{error}</Notice> : results.length ? results.map(track => <Button variant="ghost" className="h-auto w-full justify-start whitespace-normal py-3 text-start" key={track.id} onClick={() => { onChange(single ? [track] : [...value, track]); changeQuery(""); if (single) setOpen(false); }}><div><div>{track.title}</div><div className="text-sm text-muted-foreground">{track.artist || "Unknown artist"} · {track.album || "Unknown album"}</div></div></Button>) : <p className="p-4 text-sm text-muted-foreground">No matching unselected tracks.</p>}</div></ScrollArea></DialogContent></Dialog>
  </div>;
}
