import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./sidebarCurations.ts', import.meta.url), 'utf8'))
  .replace(/^import .* from "react";$/m, '')
  .replace(/^export /gm, '');

function setup(onboarding = false) {
  let state, cleanup, dependencies, resolveRequest, requests = 0;
  const listeners = new Map();
  const window = {
    addEventListener: (name, handler) => listeners.set(name, handler),
    removeEventListener: name => listeners.delete(name),
    dispatchEvent: event => listeners.get(event.type)?.(event),
  };
  const hooks = runInNewContext(`${source}\n({ useSidebarCurations, publishCurations })`, {
    window, AbortController,
    CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } },
    useState: initial => { state = initial; return [state, value => { state = value; }]; },
    useEffect: (effect, deps) => { dependencies = deps; cleanup = effect(); },
    fetch: () => { requests++; return new Promise(resolve => { resolveRequest = resolve; }); },
  });
  hooks.useSidebarCurations(onboarding);
  return {
    ...hooks, listeners, get state() { return state; }, get requests() { return requests; },
    get dependencies() { return dependencies; }, cleanup: () => cleanup?.(),
    finish: async curations => {
      resolveRequest({ ok: true, json: async () => ({ curations }) });
      await new Promise(resolve => setImmediate(resolve));
    },
  };
}

test('sidebar loads once with no pathname dependency and accepts curation updates', async () => {
  const hook = setup();
  const initial = [{ id: 'one', name: 'One', tracks: [] }];
  await hook.finish(initial);
  assert.equal(hook.requests, 1);
  assert.equal(JSON.stringify(hook.dependencies), '[false]');
  assert.equal(hook.state, initial);
  const updated = [{ id: 'two', name: 'Renamed', tracks: [1] }];
  hook.publishCurations(updated);
  assert.equal(hook.state, updated);
  hook.publishCurations([]);
  assert.equal(hook.state.length, 0);
  assert.equal(hook.requests, 1);
  hook.cleanup();
  assert.equal(hook.listeners.size, 0);
});

test('a late initial response cannot overwrite a saved or deleted curation', async () => {
  const hook = setup();
  const updated = [{ id: 'new', name: 'New', tracks: [] }];
  hook.publishCurations(updated);
  await hook.finish([{ id: 'old', name: 'Old', tracks: [] }]);
  assert.equal(hook.state, updated);
  hook.cleanup();
});

test('cleanup ignores late responses and onboarding never fetches curations', async () => {
  const hook = setup();
  hook.cleanup();
  await hook.finish([{ id: 'old', name: 'Old', tracks: [] }]);
  assert.equal(hook.state.length, 0);
  assert.equal(setup(true).requests, 0);
});

test('authenticated pages share one shell layout without changing their URLs', () => {
  const layout = readFileSync(new URL('../../app/(authenticated)/layout.tsx', import.meta.url), 'utf8');
  assert.match(layout, /<AuthenticatedShell>\{children\}<\/AuthenticatedShell>/);
  for (const file of ['HomeGrid', 'browse/BrowseLibrary', 'curate/CurateLibrary', 'settings/SettingsView', 'map/MusicGalaxy', 'sync/SyncLibrary', 'artists/ArtistProfileView']) {
    const page = readFileSync(new URL(`../${file}.tsx`, import.meta.url), 'utf8');
    assert.doesNotMatch(page, /<AppShell/);
  }
});
