import { emptyMetadataFilters, type MetadataFilters } from "./metadataFilters";

/** Library search and filters as they appear in the address bar. */
export type LibrarySort = "name" | "artist" | "released" | "date_added";
export type LibraryUrlState = { q: string; artist: string; album: string; sortBy: LibrarySort; sortDirection: "asc" | "desc"; metadata: MetadataFilters };

const sorts: LibrarySort[] = ["name", "artist", "released", "date_added"];
const listKeys = ["vocals", "language", "lyrics", "translation", "genre"] as const;

export function parseLibraryUrl(search: string): LibraryUrlState {
  const params = new URLSearchParams(search);
  const sort = params.get("sort") as LibrarySort | null;
  const metadata: MetadataFilters = { ...emptyMetadataFilters };
  for (const key of listKeys) metadata[key] = params.getAll(key).filter(Boolean);
  metadata.year_from = params.get("year_from") || "";
  metadata.year_to = params.get("year_to") || "";
  return {
    q: params.get("q") || "", artist: params.get("artist") || "", album: params.get("album") || "",
    sortBy: sort && sorts.includes(sort) ? sort : "name",
    sortDirection: params.get("order") === "desc" ? "desc" : "asc",
    metadata,
  };
}

/** The query string for a state, leaving out defaults so a plain library stays `/library`. */
export function libraryUrlSearch(state: LibraryUrlState): string {
  const params = new URLSearchParams();
  if (state.q) params.set("q", state.q);
  if (state.artist) params.set("artist", state.artist);
  if (state.album) params.set("album", state.album);
  if (state.sortBy !== "name") params.set("sort", state.sortBy);
  if (state.sortDirection !== "asc") params.set("order", state.sortDirection);
  for (const key of listKeys) for (const value of state.metadata[key]) params.append(key, value);
  if (state.metadata.year_from) params.set("year_from", state.metadata.year_from);
  if (state.metadata.year_to) params.set("year_to", state.metadata.year_to);
  const text = params.toString();
  return text ? `?${text}` : "";
}
