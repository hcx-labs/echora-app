export type MetadataFilters = { vocals: string[]; language: string[]; lyrics: string[]; translation: string[]; genre: string[]; year_from: string; year_to: string };
export const emptyMetadataFilters: MetadataFilters = { vocals: [], language: [], lyrics: [], translation: [], genre: [], year_from: "", year_to: "" };
export function hasMetadataFilters(filters: MetadataFilters) { return Object.values(filters).some(value => value.length > 0); }
export function appendMetadataFilters(params: URLSearchParams, filters: MetadataFilters) {
  for (const [key, value] of Object.entries(filters)) {
    if (Array.isArray(value)) for (const item of value) params.append(key, item);
    else if (value) params.set(key, value);
  }
}
