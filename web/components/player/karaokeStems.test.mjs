import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

const source = stripTypeScriptTypes(readFileSync(new URL('./karaokeStems.ts', import.meta.url), 'utf8'));
const { stemGains, StemMixer } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('the balance runs from vocals only through the full song to instrumental only', () => {
  assert.deepEqual(stemGains(-1), { vocals: 1, instrumental: 0 });
  assert.deepEqual(stemGains(0), { vocals: 1, instrumental: 1 });
  assert.deepEqual(stemGains(1), { vocals: 0, instrumental: 1 });
  assert.deepEqual(stemGains(0.25), { vocals: 0.75, instrumental: 1 });
  assert.deepEqual(stemGains(5), { vocals: 0, instrumental: 1 });
  assert.deepEqual(stemGains(Number.NaN), { vocals: 1, instrumental: 1 });
});

function fakeAudio() {
  const log = [];
  class Node {
    constructor(name) { this.name = name; }
    connect(target, output, input) { log.push([this.name, target.name, output, input]); return target; }
    disconnect() { log.push([this.name, 'disconnect']); }
  }
  class Param { constructor() { this.value = 1; } cancelScheduledValues() {} setValueAtTime(v) { this.value = v; } setTargetAtTime(v) { this.value = v; } }
  let gains = 0, mergers = 0;
  class Context {
    constructor() { this.state = 'suspended'; this.currentTime = 0; this.destination = new Node('out'); }
    createMediaElementSource() { return new Node('source'); }
    createChannelSplitter(count) { assert.equal(count, 4); return new Node('split'); }
    createChannelMerger() { return new Node(`merge${mergers++}`); }
    createGain() { const node = new Node(gains++ ? 'instrumental' : 'vocals'); node.gain = new Param(); return node; }
    resume() { this.state = 'running'; return Promise.resolve(); }
  }
  globalThis.AudioContext = Context;
  return { log, gain: () => null };
}

test('stems split into vocal and instrumental pairs; the original goes straight out', () => {
  const { log } = fakeAudio();
  const element = {};
  const mixer = new StemMixer(element);
  assert.equal(mixer.active, false);
  mixer.route(false);
  assert.equal(mixer.active, false, 'the original song needs no graph');
  mixer.route(true);
  assert.equal(element.crossOrigin, 'use-credentials');
  assert.deepEqual(log.filter(([from]) => from === 'split'), [
    ['split', 'merge0', 0, 0], ['split', 'merge0', 1, 1],
    ['split', 'merge1', 2, 0], ['split', 'merge1', 3, 1],
  ]);
  assert.deepEqual(log.at(-1), ['source', 'split', undefined, undefined]);
  mixer.route(false);
  assert.deepEqual(log.at(-1), ['source', 'out', undefined, undefined]);
  delete globalThis.AudioContext;
});
