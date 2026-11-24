import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const load = name => `data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(readFileSync(new URL(name, import.meta.url), 'utf8'))).toString('base64')}`;
const filters = await import(load('./metadataFilters.ts'));
const source = readFileSync(new URL('./libraryUrl.ts', import.meta.url), 'utf8').replace(/from "\.\/metadataFilters"/, `from "${load('./metadataFilters.ts')}"`);
const { parseLibraryUrl, libraryUrlSearch } = await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString('base64')}`);

test('a plain library has a clean URL', () => {
  assert.equal(libraryUrlSearch(parseLibraryUrl('')), '');
  assert.deepEqual(parseLibraryUrl('').metadata, filters.emptyMetadataFilters);
});

test('search, filters and sort round-trip through the URL', () => {
  const state = {
    q: 'night drive', artist: 'Bo Baskoro', album: 'Love You Now', sortBy: 'released', sortDirection: 'desc',
    metadata: { ...filters.emptyMetadataFilters, genre: ['Pop', 'R&B'], language: ['ko'], year_from: '2010', year_to: '2020' },
  };
  const search = libraryUrlSearch(state);
  assert.match(search, /^\?q=night\+drive&artist=Bo\+Baskoro/);
  assert.deepEqual(parseLibraryUrl(search), state);
});

test('unknown sort values fall back to defaults', () => {
  const state = parseLibraryUrl('?sort=bogus&order=sideways');
  assert.equal(state.sortBy, 'name');
  assert.equal(state.sortDirection, 'asc');
});
