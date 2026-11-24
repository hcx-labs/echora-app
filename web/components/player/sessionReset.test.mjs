import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

// Execute the provider's real reset handler with browser/state adapters. This
// covers teardown behavior without pretending to render a React component.
const source = readFileSync(new URL("./PlayerProvider.tsx", import.meta.url), "utf8");
const handler = source.match(/const sessionExpired = \(\) => \{([\s\S]*?)\n    \};/);
assert.ok(handler, "session reset handler exists");

test("session expiry clears account data, stops audio, and invalidates pending work", () => {
  const state = {};
  const events = [];
  const calls = [];
  const ref = current => ({ current });
  const context = {
    playbackSessionRef: ref(4), trackRef: ref({ id: "old" }),
    queueRef: ref([{ id: "old" }]), queueIndexRef: ref(0),
    lyricsCacheRef: ref(new Map([["old", { text: "private" }]])),
    lyricsGenerationRef: ref(new Map([["old", 1]])),
    paletteRef: ref({ accent: [1, 2, 3] }), paletteTrackRef: ref("old"),
    visualRequest: ref({ abort: () => calls.push("abort") }), visualFeatures: ref({}),
    listenedRef: ref(10), announcedRef: ref("old"), analysisFrame: ref(12),
    player: { pause: () => calls.push("pause"), removeAttribute: value => calls.push(value), load: () => calls.push("load") },
    cancelAnimationFrame: value => calls.push(value), resetVisuals: () => calls.push("reset"),
    publishPlaybackTime: value => state.time = value, publishPalette: value => state.palette = value,
    window: { dispatchEvent: event => events.push(event) },
    CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
    hasMediaSession: () => true,
    navigator: { mediaSession: { metadata: {}, playbackState: "playing", setPositionState: () => calls.push("position") } },
  };
  for (const name of ["Track", "Queue", "QueueIndex", "Lyrics", "LyricsLoading", "AudioQuality", "WaveformData", "CurrentTime", "Duration", "Buffered", "Playing", "Buffering", "Expanded"]) {
    context[`set${name}`] = value => state[name] = value;
  }
  vm.runInNewContext(`(() => {${handler[1]}})()`, context);
  assert.equal(context.playbackSessionRef.current, 5);
  assert.equal(context.trackRef.current, null);
  assert.equal(context.queueRef.current.length, 0);
  assert.equal(context.queueIndexRef.current, -1);
  assert.equal(context.lyricsCacheRef.current.size, 0);
  assert.equal(context.lyricsGenerationRef.current.size, 0);
  assert.equal(context.visualRequest.current, null);
  assert.equal(context.visualFeatures.current, null);
  assert.equal(context.analysisFrame.current, 0);
  for (const name of ["Track", "Lyrics", "AudioQuality", "WaveformData"]) assert.equal(state[name], null);
  for (const name of ["Playing", "Buffering", "Expanded", "LyricsLoading"]) assert.equal(state[name], false);
  assert.equal(state.Queue.length, 0);
  assert.equal(state.time, 0);
  assert.equal(context.navigator.mediaSession.metadata, null);
  assert.equal(context.navigator.mediaSession.playbackState, "none");
  for (const call of ["pause", "src", "load", "abort", 12, "position"]) assert.ok(calls.includes(call));
  assert.ok(events.some(event => event.type === "echora:track-change" && event.detail === null));
});
