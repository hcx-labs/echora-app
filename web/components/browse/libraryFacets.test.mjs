import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

const source = stripTypeScriptTypes(readFileSync(new URL("./libraryFacets.ts", import.meta.url), "utf8"));
const { loadLibraryFacets } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

test("an old facet response cannot replace newer results or finish their loading state", async t => {
  const responses = [deferred(), deferred()];
  let request = 0;
  t.mock.method(globalThis, "fetch", () => responses[request++].promise);
  const old = new AbortController();
  const current = new AbortController();
  const published = [];
  let finished = 0;
  const load = controller => loadLibraryFacets(new URLSearchParams(), controller.signal, value => published.push(value), () => finished++);
  const first = load(old);
  old.abort();
  const second = load(current);
  const latest = { artists: [{ name: "New", tracks: 1 }], albums: [] };
  responses[1].resolve({ ok: true, json: async () => latest });
  await second;
  responses[0].resolve({ ok: true, json: async () => ({ artists: [{ name: "Old", tracks: 2 }], albums: [] }) });
  await first;
  assert.deepEqual(published, [latest]);
  assert.equal(finished, 1);
});

test("aborting during JSON decoding also prevents publication", async t => {
  const body = deferred();
  t.mock.method(globalThis, "fetch", async () => ({ ok: true, json: () => body.promise }));
  const controller = new AbortController();
  let updates = 0;
  const pending = loadLibraryFacets(new URLSearchParams(), controller.signal, () => updates++, () => updates++);
  await Promise.resolve();
  controller.abort();
  body.resolve({ artists: [], albums: [] });
  await pending;
  assert.equal(updates, 0);
});
