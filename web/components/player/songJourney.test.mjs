import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
const load = file => import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(readFileSync(new URL(file, import.meta.url), 'utf8'))).toString('base64')}`);
const { planSongSections, sectionAt } = await load('./visualFeatures.ts');
const { SongJourney, gradeColor } = await load('./songJourney.ts');

// Bins of 2 s: verse (A), chorus (B), verse (A) again. Self-similarity is high within a
// part and between the two verses, low across verse and chorus.
function structure(parts) {
  const labels = parts.flatMap(([label, bins]) => Array(bins).fill(label));
  const n = labels.length;
  const matrix = labels.map(a => labels.map(b => a === b ? .9 : .2));
  return { edges_seconds: Array.from({ length: n + 1 }, (_, i) => i * 2), novelty: Array(n).fill(.3), chroma_cosine: matrix, mfcc_rbf: matrix };
}
const hop = 512 / 22050;
function levels(parts) {
  return parts.flatMap(([, bins, level]) => Array(Math.round(bins * 2 / hop)).fill(level));
}

test('sections follow the structure, a returning part returns to its scene, energy ranks the chorus highest', () => {
  const parts = [['A', 12, .2], ['B', 12, .8], ['A', 12, .25]];
  const sections = planSongSections(structure(parts), levels(parts), hop, 72);
  assert.deepEqual(sections.map(section => section.start), [0, 24, 48]);
  assert.equal(sections.at(-1).end, 72);
  assert.deepEqual(sections.map(section => section.scene), [0, 1, 0]);
  assert.equal(Math.max(...sections.map(section => section.energy)), sections[1].energy);
  assert.ok(sections[0].energy < sections[2].energy);
});

test('songs without usable structure are one section', () => {
  assert.deepEqual(planSongSections(undefined, [.1], hop, 30), [{ start: 0, end: 30, scene: 0, energy: .5 }]);
  const short = structure([['A', 4, .5]]);
  assert.equal(planSongSections(short, levels([['A', 4, .5]]), hop, 8).length, 1);
});

test('sectionAt finds the section and the next section energy', () => {
  const sections = [{ start: 0, end: 10, scene: 0, energy: .2 }, { start: 10, end: 20, scene: 1, energy: .9 }];
  assert.deepEqual(sectionAt(sections, 9.9), { index: 0, count: 2, scene: 0, energy: .2, nextEnergy: .9, start: 0, end: 10 });
  assert.equal(sectionAt(sections, 15).nextEnergy, null);
  assert.equal(sectionAt([], 3), null);
});

const sensitivity = { bassReactivity: 1, vocalReactivity: 1, trebleReactivity: 1 };
const sections = [{ start: 0, end: 20, scene: 0, energy: .1 }, { start: 20, end: 40, scene: 1, energy: 1 }];
function frame(time, extra = {}) {
  return { active: true, trackId: 'song', timestamp: time, bpm: 120, beat: false, bass: .4, mid: .4, treble: .2,
    section: sectionAt(sections, time), features: null, source: { durationSeconds: 40 }, enrichment: null, ...extra };
}
function play(journey, from, to) {
  for (let time = from; time < to; time += 1 / 30) { journey.receive(frame(time), sensitivity, time); journey.step(1 / 30, 1, time); }
}

test('a build-up rises before a bigger section and the impact lands on the boundary', () => {
  const journey = new SongJourney();
  play(journey, 0, 12);
  assert.ok(journey.anticipation < .05, 'no build-up far from the boundary');
  play(journey, 12, 19.9);
  assert.ok(journey.anticipation > .4, `build-up before the chorus: ${journey.anticipation}`);
  play(journey, 19.9, 20.1);
  assert.ok(journey.impact > .6, `impact on the boundary: ${journey.impact}`);
  assert.equal(journey.scene, 1);
  assert.ok(journey.sceneBlend < .2, 'the new scene blends in');
  play(journey, 20.1, 26);
  assert.ok(journey.impact < .01 && journey.sceneBlend === 1);
  assert.ok(journey.travel > 0);
});

test('seeking into a new section changes scene without an impact', () => {
  const journey = new SongJourney();
  play(journey, 0, 2);
  play(journey, 30, 31);
  assert.equal(journey.scene, 1);
  assert.equal(journey.sceneBlend, 1);
  assert.ok(journey.impact < .01);
});

test('pausing eases the journey to rest, and a new track starts fresh', () => {
  const journey = new SongJourney();
  play(journey, 0, 10);
  // Motion continues for the 0.7 s grace period after the last frame, then stops.
  const pause = (from, to) => { for (let i = from; i < to; i++) { journey.receive({ ...frame(10), active: false }, sensitivity, 10 + i / 30); journey.step(1 / 30, 1, 10 + i / 30); } };
  pause(0, 30);
  const travelled = journey.travel;
  pause(30, 300);
  assert.ok(journey.drive < .01);
  assert.equal(journey.travel, travelled);
  journey.receive(frame(0, { trackId: 'next' }), sensitivity, 30);
  assert.equal(journey.travel, 0);
  assert.equal(journey.beats, 0);
});

test('beats reset the beat phase, which then advances with the tempo', () => {
  const journey = new SongJourney();
  journey.receive(frame(1, { beat: true }), sensitivity, 1);
  assert.equal(journey.beatPhase, 0);
  assert.equal(journey.beatPulse, 1);
  journey.step(.1, 1, 1.1);
  assert.ok(Math.abs(journey.beatPhase - .2) < 1e-9);
});

test('grading keeps a colour at rest and rotates hue without changing lightness much', () => {
  const color = [.8, .4, .3];
  gradeColor(color, 0).forEach((value, index) => assert.ok(Math.abs(value - color[index]) < .01));
  const turned = gradeColor(color, Math.PI / 2);
  assert.ok(Math.abs(turned[0] - color[0]) > .1);
  const journey = new SongJourney();
  assert.equal(journey.color([color], 0).length, 3);
});
