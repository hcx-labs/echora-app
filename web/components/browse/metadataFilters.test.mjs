import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const source = stripTypeScriptTypes(readFileSync(new URL('./metadataFilters.ts', import.meta.url), 'utf8'));
const { emptyMetadataFilters, hasMetadataFilters, appendMetadataFilters } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
test('empty arrays do not enable filters or send empty parameters', () => {
  assert.equal(hasMetadataFilters(emptyMetadataFilters), false);
  const params = new URLSearchParams(); appendMetadataFilters(params, emptyMetadataFilters);
  assert.equal(params.toString(), '');
});
test('every selected tag is sent as a repeated query value without splitting genre names', () => {
  const filters = {...emptyMetadataFilters, vocals:['vocal','instrumental'], lyrics:['available','missing'], language:['en','ja'], translation:['en','de'], genre:['Jazz, Blues','R&B'], year_from:'2000'};
  const params = new URLSearchParams(); appendMetadataFilters(params, filters);
  for(const key of ['vocals','lyrics','language','translation','genre']) assert.deepEqual(new URLSearchParams(params.toString()).getAll(key), filters[key]);
  assert.equal(params.get('year_from'),'2000'); assert.equal(hasMetadataFilters(filters),true);
});
