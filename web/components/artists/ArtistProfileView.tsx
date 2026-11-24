"use client";

import { ArrowLeft, Disc3 } from "lucide-react";
import { Button } from "../ui/button";
import { LoadingState } from "../ui/spinner";
import { useEffect, useState } from "react";
import TransitionLink from "../shell/TransitionLink";
import { Workspace, WorkspaceBody } from "../layout/workspace";
import { PageHeader } from "../layout/page-header";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../ui/tabs";
import LoadingImage from "../media/LoadingImage";
import { coverArtUrl } from "../media/coverArt";

type Track = { id: string; title: string; artist?: string; album?: string };
type Facet = { index: number; weight: number; track_count: number; representative_track: Track };
type Profile = { artist: string; model: string; track_count: number; component_count: number; facets: Facet[] };
type Similar = Profile & { similarity: number; target_coverage: number; candidate_coverage: number; strongest_facet_match: { target_facet: number; candidate_facet: number; similarity: number } };

export default function ArtistProfileView({ artist }: { artist: string }) {
  const [model, setModel] = useState("muq_mulan");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [similar, setSimilar] = useState<Similar[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [artwork, setArtwork] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    queueMicrotask(() => { if (!controller.signal.aborted) setArtwork(""); });
    Promise.all([
      fetch("/analysis/auth/me", { signal: controller.signal }).then(response => response.ok ? response.json() : null),
      fetch(`/analysis/library/tracks?${new URLSearchParams({ artist, limit: "20" })}`, { signal: controller.signal }).then(response => response.ok ? response.json() : null),
    ]).then(([user, library]) => {
      const track = library?.tracks?.find((item: { cover_art?: string }) => item.cover_art);
      if (!controller.signal.aborted && user?.navidrome_connection_id && track?.cover_art) setArtwork(coverArtUrl(user.navidrome_connection_id, track.cover_art, 1024));
    }).catch(() => {});
    return () => controller.abort();
  }, [artist]);

  useEffect(() => {
    const controller = new AbortController();
    queueMicrotask(() => { if (!controller.signal.aborted) { setLoading(true); setError(""); } });
    const query = new URLSearchParams({ artist, model });
    Promise.all([
      fetch(`/analysis/library/artists/profile?${query}`, { signal: controller.signal }).then(response => response.ok ? response.json() : Promise.reject(new Error("Artist profile unavailable"))),
      fetch(`/analysis/library/artists/similar?${query}`, { signal: controller.signal }).then(response => response.ok ? response.json() : Promise.reject(new Error("Artist similarities unavailable"))),
    ]).then(([nextProfile, nextSimilar]) => { if (!controller.signal.aborted) { setProfile(nextProfile); setSimilar(nextSimilar.results || []); } })
      .catch(reason => { if (reason.name !== "AbortError") setError(reason.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [artist, model]);

  return <>
    <Workspace>
      <PageHeader eyebrow="Artist" title={artist} description="The distinct sounds in this catalogue, and the artists connected to them." actions={<Button variant="outline" asChild><TransitionLink href="/library"><ArrowLeft />Library</TransitionLink></Button>} />
      <WorkspaceBody>
      <section aria-label="Artist overview" className="grid gap-6 border-b border-border pb-8 sm:grid-cols-[180px_minmax(0,1fr)] sm:items-end">
        <div className="relative aspect-square w-full max-w-[180px] overflow-hidden rounded-none bg-muted">{artwork ? <LoadingImage src={artwork} alt={`Library artwork for ${artist}`} sizes="180px" /> : <div className="flex h-full flex-col items-center justify-center gap-4 text-muted-foreground"><Disc3 className="size-20" strokeWidth={1} /><span className="text-sm">{loading ? "Loading artwork…" : "No library artwork available"}</span></div>}</div>
        <div className="flex flex-col justify-end gap-6">{!loading && !error && profile && <dl className="flex flex-wrap gap-10"><div><dt className="text-sm text-muted-foreground">Analyzed tracks</dt><dd className="mt-1 text-[28px] leading-tight font-semibold tabular-nums">{profile.track_count}</dd></div><div><dt className="text-sm text-muted-foreground">Sound facets</dt><dd className="mt-1 text-[28px] leading-tight font-semibold tabular-nums">{profile.component_count}</dd></div></dl>}</div>
      </section>
      <Tabs value={model} onValueChange={setModel} className="gap-6 py-8">
        <div className="flex flex-wrap items-center justify-between gap-4"><TabsList aria-label="Artist representation"><TabsTrigger value="muq_mulan">Semantic</TabsTrigger><TabsTrigger value="mert">Acoustic</TabsTrigger></TabsList><p className="text-sm text-muted-foreground">{model === "mert" ? "Matching acoustic texture and sound" : "Matching musical meaning and character"}</p></div>
        {["muq_mulan", "mert"].map(value => <TabsContent key={value} value={value}>
          {loading ? <LoadingState label="Loading catalogue facets and similar artists…" /> : error ? <p role="alert" className="py-8 text-destructive">{error}</p> : <div className="grid gap-10 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
            <section aria-labelledby={`facets-${value}`}><h2 id={`facets-${value}`} className="mb-2 text-base font-semibold">Catalogue facets</h2><div className="space-y-4">{profile?.facets.map(facet => <article key={facet.index} className="space-y-4 border-b border-border py-5"><div className="space-y-2"><p className="text-sm text-muted-foreground">Facet {facet.index + 1} • {Math.round(facet.weight * 100)}% of this representation</p><h3 className="text-base font-medium">{facet.representative_track.title}</h3></div><dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-6 gap-y-2 text-sm"><dt className="text-muted-foreground">Album</dt><dd>{facet.representative_track.album || "Unknown album"}</dd><dt className="text-muted-foreground">Tracks</dt><dd>{facet.track_count}</dd></dl></article>)}{!profile?.facets.length && <p className="text-sm text-muted-foreground">No analyzed facets are available for this artist.</p>}</div></section>
            <section aria-labelledby={`similar-${value}`}><h2 id={`similar-${value}`} className="text-base font-semibold">Similar artists</h2><p className="mb-5 mt-2 text-sm text-muted-foreground">Similarity compares sound facets in both directions.</p>{!similar.length ? <p className="py-8 text-sm text-muted-foreground">No similar artists found for this representation.</p> : <ul className="divide-y divide-border">{similar.map(item => <li key={item.artist}><TransitionLink href={`/artists/${encodeURIComponent(item.artist)}`} className="flex items-start justify-between gap-4 rounded-none py-5 focus-visible:outline-2 focus-visible:outline-ring"><div className="min-w-0"><h3 className="break-words font-medium text-foreground">{item.artist}</h3><p className="mt-1 text-sm text-muted-foreground">{item.track_count} tracks • {item.component_count} facets</p><dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 break-words text-xs text-muted-foreground"><dt>Catalogue coverage</dt><dd>{Math.round(item.target_coverage * 100)}% / {Math.round(item.candidate_coverage * 100)}%</dd><dt>Strongest match</dt><dd>Facet {item.strongest_facet_match.target_facet + 1} to {item.strongest_facet_match.candidate_facet + 1} • {Math.round(item.strongest_facet_match.similarity * 100)}%</dd></dl></div><div className="shrink-0 rounded-none border border-border bg-muted px-3 py-2 text-right"><p className="text-lg font-semibold tabular-nums text-primary">{Math.round(item.similarity * 100)}%</p><p className="text-xs text-muted-foreground">similarity</p></div></TransitionLink></li>)}</ul>}</section>
          </div>}
        </TabsContent>)}
      </Tabs>
      </WorkspaceBody>
    </Workspace>
  </>;
}
