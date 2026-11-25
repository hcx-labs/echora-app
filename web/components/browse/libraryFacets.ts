export type LibraryFacet = { name: string; tracks: number };
type Facets = { artists: LibraryFacet[]; albums: LibraryFacet[] };

/** An aborted request must not publish results or finish a newer loading state. */
export async function loadLibraryFacets(
  params: URLSearchParams,
  signal: AbortSignal,
  publish: (facets: Facets) => void,
  finish: () => void,
): Promise<void> {
  try {
    const response = await fetch(`/analysis/library/facets?${params}`, { signal });
    const body: Facets | null = response.ok ? await response.json() : null;
    if (!signal.aborted && body) publish(body);
  } catch {
    // Keep existing facets if the server is unavailable.
  } finally {
    if (!signal.aborted) finish();
  }
}
